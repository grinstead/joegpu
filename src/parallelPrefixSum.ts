import { COMMON_DEFS } from "./gpu_utils.ts";

function parallelPrefixSum(device: GPUDevice, maxSize: number) {
  const sumScanShader = device.createShaderModule({
    label: "Parallel Prefix Sum",
    code: `
${COMMON_DEFS}

override workgroupSize: u32 = 256;
override itemsPerThread: u32 = 2;
override tileLength: u32 = itemsPerThread * workgroupSize;

@group(0) @binding(0) var<storage, read_write> tileAccs: array<u32>;
@group(1) @binding(0) var<storage, read_write> values: array<u32>;
@group(1) @binding(0) var<uniform> slice: Slice;

var<workgroup> accumulators: array<u32, tileLength>;

/**
 * Code is port of https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */
@compute @workgroup_size(workgroupSize)
fn parallelPrefixSum(
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) workgroup: vec3u,
) {
  for (var i: u32 = localIndex; i < tileLength; i += workgroupSize) {
    let valueIndex = tileLength * workgroup.x + i;
    if (valueIndex < slice.length) {
      accumulators[i] = values[slice.offset + valueIndex];
    }
  }

  var delta: u32 = 1;
  for (
    var numThreads: u32 = tileLength / 2;
    numThreads > 0;
    numThreads /= 2
  ) {
    workgroupBarrier();

    for (var i: u32 = localIndex; i < numThreads; i += workgroupSize) {
      let lower = delta * (2 * i + 1) - 1;
      accumulators[lower] += accumulators[lower + delta];
    }

    delta *= 2;
  }

  if (localIndex == 0) {
    // record the total for this tile
    tileAccs[workgroup.x] = accumulators[tileLength - 1];

    // we set the last element to 0, that will propagate down to the first bucket
    accumulators[tileLength - 1] = 0;
  }

  for (int numThreads = 1; numThreads < tileLength; numThreads *= 2) {
    delta /= 2;

    workgroupBarrier();
    for (var i: u32 = localIndex; i < numThreads; i += workgroupSize) {
      let lower = delta * (2 * i + 1) - 1;
      let temp = accumulators[lower];
      accumulators[lower] = accumulators[lower + delta];
      accumulators[lower + delta] = accumulators[lower];
    }
  }

  workgroupBarrier();

  for (var i: u32 = localIndex; i < tileLength; i += workgroupSize) {
    let valueIndex = tileLength * workgroup.x + i;
    if (valueIndex < slice.length) {
      values[slice.offset + valueIndex] = accumulators[i];
    }
  }
}

    
`,
  });

  const combineTileValues = device.createShaderModule({
    label: "Parallel Prefix Sum",
    code: `
${COMMON_DEFS}

override tileLength: u32 = 256 * 2;

@group(0) @binding(0) var<uniform> tileAccs: array<u32>
@group(1) @binding(0) var<storage, read_write> values: array<u32>;
@group(1) @binding(0) var<uniform> slice: Slice;

/**
 * Code is port of https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */
@compute @workgroup_size(256)
fn addPriorTiles(
  @builtin(global_invocation_id) index: u32,
) {
  if (index.x >= slice.length) {
    return;
  }

  values[slice.offset + index.x] += tileAccs[index.x / tileLength];
}
`,
  });
}
