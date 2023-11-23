import { COMMON_DEFS, writeToBuffer } from "./gpu_utils.ts";
import { NUM_BYTES_UINT32 } from "./utils.ts";

const DEFAULT_WORKGROUP_SIZE = 256;
const DEFAULT_TILE_SIZE = DEFAULT_WORKGROUP_SIZE * 2;

export type PrefixSumBuffers = {
  values: GPUBuffer;
  slice: GPUBuffer;
};

export function parallelPrefixSum(device: GPUDevice, maxSize: number) {
  const sumScanShader = device.createShaderModule({
    label: "Parallel Prefix Sum",
    code: `
${COMMON_DEFS}

override workgroupSize: u32 = 256;
override tileLength: u32 = 2 * workgroupSize;

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

@compute @workgroup_size(256)
fn addGlobalOffsets(
  @builtin(global_invocation_id) index: u32,
) {
  if (index.x >= slice.length) {
    return;
  }

  values[slice.offset + index.x] += tileAccs[index.x / tileLength];
}

`,
  });

  const tileScans = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "parallelPrefixSum",
      module: sumScanShader,
    },
  });

  const maxTiles = Math.ceil(maxSize / DEFAULT_TILE_SIZE);
  const tileAccsSize =
    maxTiles <= DEFAULT_TILE_SIZE
      ? DEFAULT_TILE_SIZE
      : DEFAULT_WORKGROUP_SIZE *
        (1 << Math.ceil(Math.log2(maxTiles / DEFAULT_WORKGROUP_SIZE)));

  const crossTileScans =
    tileAccsSize === DEFAULT_TILE_SIZE
      ? tileScans
      : device.createComputePipeline({
          layout: "auto",
          compute: {
            entryPoint: "parallelPrefixSum",
            module: sumScanShader,
            constants: {
              tileLength: tileAccsSize,
            },
          },
        });

  const addGlobalOffsets = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "addGlobalOffsets",
      module: sumScanShader,
    },
  });

  const tileAccs = device.createBuffer({
    label: "Holds Tile Totals",
    size: tileAccsSize * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  const tileScansBinding = device.createBindGroup({
    layout: tileScans.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: tileAccs } }],
  });

  const tileAccsSliceBuffer = device.createBuffer({
    label: "The Slice of tileAccs that is valid",
    size: 2 * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  writeToBuffer(
    device.queue,
    tileAccsSliceBuffer,
    new Uint32Array([0, maxTiles])
  );

  const totalSumBuffer = device.createBuffer({
    label: "Holds the final total",
    size: NUM_BYTES_UINT32,
    usage: GPUBufferUsage.STORAGE,
  });

  const crossTileScansBindings = [
    device.createBindGroup({
      layout: crossTileScans.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: totalSumBuffer } }],
    }),
    device.createBindGroup({
      layout: crossTileScans.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: tileAccs } },
        { binding: 1, resource: { buffer: tileAccsSliceBuffer } },
      ],
    }),
  ];

  const addGlobalOffsetsBinding = device.createBindGroup({
    layout: addGlobalOffsets.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: tileAccs } }],
  });

  let userBindings:
    | undefined
    | {
        tileScansBinding: GPUBindGroup;
        addGlobalOffsetsBinding: GPUBindGroup;
      };

  return {
    prep(buffers: PrefixSumBuffers) {
      userBindings = {
        tileScansBinding: device.createBindGroup({
          layout: tileScans.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: buffers.values } },
            { binding: 1, resource: { buffer: buffers.slice } },
          ],
        }),
        addGlobalOffsetsBinding: device.createBindGroup({
          layout: tileScans.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: buffers.values } },
            { binding: 1, resource: { buffer: buffers.slice } },
          ],
        }),
      };
    },

    runPrefixSum,
  };

  function runPrefixSum(encoder: GPUCommandEncoder, numItems: number) {
    if (!userBindings) {
      throw new Error("Must call prep before running");
    }

    encoder.clearBuffer(tileAccs);

    const tileScansPass = encoder.beginComputePass();
    tileScansPass.setPipeline(tileScans);
    tileScansPass.setBindGroup(0, tileScansBinding);
    tileScansPass.setBindGroup(1, userBindings.tileScansBinding);
    tileScansPass.dispatchWorkgroups(Math.ceil(numItems / DEFAULT_TILE_SIZE));

    const crossTileScansPass = encoder.beginComputePass();
    crossTileScansPass.setPipeline(crossTileScans);
    crossTileScansBindings.forEach((group, i) => {
      crossTileScansPass.setBindGroup(i, group);
    });
    crossTileScansPass.dispatchWorkgroups(1);

    const addGlobalOffsetsPass = encoder.beginComputePass();
    addGlobalOffsetsPass.setPipeline(addGlobalOffsets);
    addGlobalOffsetsPass.setBindGroup(0, addGlobalOffsetsBinding);
    addGlobalOffsetsPass.setBindGroup(1, userBindings.addGlobalOffsetsBinding);
    addGlobalOffsetsPass.dispatchWorkgroups(Math.ceil(numItems / 256));
  }
}
