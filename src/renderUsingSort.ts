import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { writeToBuffer } from "./gpu_utils.ts";
import { NUM_BYTES_FLOAT32, NUM_BYTES_UINT32 } from "./utils.ts";

const CHUNK_SIZE = 16;

const TILE_SIZE = 512;
const NUM_BUCKETS = 256;

const HISTOGRAM_SIZE = NUM_BUCKETS * NUM_BYTES_UINT32;

const COMMON_DEFS = `
struct Slice {
  offset: u32,
  length: u32,
}
`;

const PROJECTED_GUASSIAN_DEF = `
struct ProjectedGaussian {
  origin: vec3f,
  // placed in here to sneak into otherwise alignment-mandated deadspace
  sortKey: u32,
  Σ_inv: vec3f,
  color: vec4f,
}
`;

const NUM_BYTES_PROJECTED_GAUSSIAN = 12 * NUM_BYTES_FLOAT32;
const NUM_BYTES_TILE = TILE_SIZE * NUM_BYTES_PROJECTED_GAUSSIAN;

export function renderUsingSort(props: GPUCanvasDetails, splatData: GPUBuffer) {
  const { canvas, context, device, format } = props;

  const chunkDims = {
    x: Math.ceil(canvas.width / CHUNK_SIZE),
    y: Math.ceil(canvas.height / CHUNK_SIZE),
  };

  const constants = [0, 1, 2, 3];
  const constantsBuffers = constants.map((c) => {
    const buffer = device.createBuffer({
      label: `Single Constant (value = ${c})`,
      size: NUM_BYTES_UINT32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    writeToBuffer(device.queue, buffer, new Uint32Array([c]));

    return buffer;
  });

  const outputTexture = device.createTexture({
    label: "Splat Output Texture",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    format: "rgba8unorm",
    size: {
      width: chunkDims.x * CHUNK_SIZE,
      height: chunkDims.y * CHUNK_SIZE,
    },
  });

  const projectGaussiansPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "projectGaussians",
      module: device.createShaderModule({
        label: "Guassian Projection Shader",
        code: `
struct GaussianSplat {
  origin: array<f32, 3>,
  normal: array<f32, 3>,
  color_sh0: array<f32, 3>,
  color_rest: array<array<f32, 3>, 15>,
  opacity: f32,
  scales: array<f32, 3>,
  quaternion: array<f32, 4>,
}

${PROJECTED_GUASSIAN_DEF}
${COMMON_DEFS}

const HARMONIC_COEFF0: f32 = 0.28209479177387814;

@group(0) @binding(0) var<storage> gaussians: array<GaussianSplat>;
@group(0) @binding(1) var<uniform> camera: mat4x4f; 
@group(0) @binding(2) var<storage, read_write> projectedGaussians: array<ProjectedGaussian>;
@group(0) @binding(3) var<uniform> slice: Slice;

fn invert_2x2(input: mat2x2<f32>) -> mat2x2<f32> {
  return (1 / determinant(input)) * mat2x2<f32>(
    input[1][1], -input[0][1],
    -input[1][0], input[0][0],
  );
}

fn normalize_opacity(in: f32) -> f32 {
  if (in >= 0) {
    return 1 / (1 + exp(-in));
  } else {
    let temp = exp(in);
    return temp / (1 + temp);
  }
}

// todo: make this overridable maybe?
const chunkDims = vec2f(16. / 512, 16. / 512);
const chunksPerRow = u32(1. / chunkDims.y);

override blockSize: u32 = 256;
@compute @workgroup_size(blockSize)
fn projectGaussians(
  @builtin(global_invocation_id) index: vec3u,
) {
  if (index.x >= slice.length) {
    return;
  }

  let in = gaussians[slice.offset + index.x];

  let camera_space_origin = camera * vec4<f32>(in.origin[0], in.origin[1], in.origin[2], 1.0);
  let z = camera_space_origin.z;

  // quaternion to matrix formula taken from
  // https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
  let qr = in.quaternion[0];
  let qi = in.quaternion[1];
  let qj = in.quaternion[2];
  let qk = in.quaternion[3];

  // R (rotation) and S (scales) matrices from Gaussian Splat Paper
  // technically these are the transposed versions because the gpu is col-major order
  let SR_T = mat4x4<f32>(
    exp(in.scales[0]), 0, 0, 0,
    0, exp(in.scales[1]), 0, 0,
    0, 0, exp(in.scales[2]), 0,
    0, 0, 0, 0,
  ) * mat4x4<f32>(
    1 - 2*(qj*qj + qk*qk), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0,
    2*(qi*qj + qk*qr), 1 - 2*(qi*qi + qk*qk), 2*(qj*qk - qi*qr), 0,
    2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi*qi + qj*qj), 0,
    0, 0, 0, 0,
  );

  // Σ is from Gaussian Splat paper (section 4, eq. 6)
  let Σ = transpose(SR_T) * SR_T;

  let JW = mat4x4<f32>(
    1 / z, 0, 0, 0,
    0, 1 / z, 0, 0,
    -camera_space_origin.x / z / z, -camera_space_origin.y / z / z, 0, 0,
    0, 0, 0, 0,
  ) * camera; 

  // x in camera space -> x coordinate in screen space
  // x as is for now, but z^-1 so derivative is -z^-2

  let Σ_prime_full = JW * Σ * transpose(JW);
  let Σ_prime_inv = invert_2x2(
    mat2x2<f32>(
      Σ_prime_full[0][0], Σ_prime_full[0][1],
      Σ_prime_full[1][0], Σ_prime_full[1][1],
    )
  );

  let screenSpace = vec3f(camera_space_origin.xy / z, z);
  let chunkId = vec2u(
    saturate((screenSpace.xy + 1) / 2) / chunkDims
  );

  projectedGaussians[slice.offset + index.x] = ProjectedGaussian(
    screenSpace,
    insertBits(
      1 + min(u32(max(0, screenSpace.z) * (1 << 13)), (1 << 16) - 2),
      min(chunkId.y, 31) * chunksPerRow + min(chunkId.x, 31),
      16,
      16,
    ),
    vec3f(Σ_prime_inv[0][0], Σ_prime_inv[0][1], Σ_prime_inv[1][1]),
    vec4<f32>(
      vec3f(in.color_sh0[0], in.color_sh0[1], in.color_sh0[2]) * HARMONIC_COEFF0 + .5,
      normalize_opacity(in.opacity),
    ),
  );
}
        `,
      }),
    },
  });

  const binSizingPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "computeBinSizes",
      module: device.createShaderModule({
        label: "Radix Sort Histogram Computation",
        code: `
${PROJECTED_GUASSIAN_DEF}
${COMMON_DEFS}

const NUM_BITS_PER_BUCKET: u32 = 8;
const NUM_PASSES: u32 = 32 / NUM_BITS_PER_BUCKET;
const NUM_BUCKETS: u32 = 1 << NUM_BITS_PER_BUCKET;

@group(0) @binding(0) var<storage> gaussians: array<ProjectedGaussian>;
@group(0) @binding(1) var<uniform> slice: Slice;

// these should always be empty
@group(0) @binding(2) var<storage, read_write> globalHistogram: array<array<atomic<u32>, NUM_BUCKETS>, NUM_PASSES>;
@group(0) @binding(3) var<storage, read_write> nextTileStart: atomic<u32>;

override blockSize: u32 = 256;
override itemsPerThreadPerTile: u32 = 16;
override tileLength: u32 = blockSize * itemsPerThreadPerTile;

var<workgroup> workgroupStart: u32;
var<workgroup> localHistogram: array<array<atomic<u32>, NUM_BUCKETS>, NUM_PASSES>;

// currently do nothing
@compute @workgroup_size(blockSize)
fn computeBinSizes(
  @builtin(local_invocation_index) localIndex: u32,
) {  
  // loop through "tiles"
  loop {
    if (localIndex == 0) {
      workgroupStart = atomicAdd(&nextTileStart, tileLength);
    }

    let tileStart = workgroupUniformLoad(&workgroupStart);
    if (tileStart >= slice.length) {
      break;
    }

    let end = min(slice.length, tileStart + tileLength);
    for (
      var i: u32 = tileStart + localIndex;
      i < end;
      i += blockSize
    ) {
      let key = gaussians[slice.offset + i].sortKey;
      
      for (var round: u32 = 0; round < NUM_PASSES; round++) {
        let subkey = extractBits(
          select(0xFFFFFFFF, key, key != 0),
          round * NUM_BITS_PER_BUCKET,
          NUM_BITS_PER_BUCKET
        );
        atomicAdd(&localHistogram[round][subkey], 1);
      }
    }

    // any space in the tile after the end of the gaussians will be treated as having
    // a 0 sort key (which gets sorted to the end of the array)
    if (end < tileStart + tileLength && localIndex < 4) {
      // the first 4 threads in the block write simultaneously
      atomicAdd(&localHistogram[localIndex][0xFF], tileStart + tileLength - end);
    }
  }

  workgroupBarrier();

  // add to the global histogram at the end
  if (localIndex == 0) {
    for (var round: u32 = 0; round < NUM_PASSES; round++) {
      for (var bucket: u32 = 0; bucket < NUM_BUCKETS; bucket++) {
        atomicAdd(
          &globalHistogram[round][bucket], 
          atomicLoad(&localHistogram[round][bucket])
        );
      }
    }
  }
}
  `,
      }),
    },
  });

  const prefixSumPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "exclusivePrefixSum",
      module: device.createShaderModule({
        label: "Radix Sort Histogram Computation",
        code: `
const NUM_PASSES = 4;
const NUM_BUCKETS = 256;

@group(0) @binding(0) var<storage, read_write> globalHistogram: array<array<u32, NUM_BUCKETS>, NUM_PASSES>;

var<workgroup> scratchSpace: array<u32, NUM_BUCKETS>;

@compute @workgroup_size(NUM_BUCKETS)
fn exclusivePrefixSum(
  @builtin(local_invocation_index) index: u32,
  @builtin(workgroup_id) group: vec3u,
) {
  let original = globalHistogram[group.x][index];
  var sum = original;
  for (var i: u32 = 1; i < NUM_BUCKETS; i *= 2) {
    scratchSpace[index] = sum;

    workgroupBarrier();

    if (index >= i) {
      sum += scratchSpace[index - i];
    }
  }

  globalHistogram[group.x][index] = sum - original;
}
        `,
      }),
    },
  });

  const histogramBuffer = device.createBuffer({
    size: 4 /* number of passes */ * NUM_BUCKETS * NUM_BYTES_UINT32,
    usage:
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.STORAGE,
  });
  const nextTileIndexBuffer = device.createBuffer({
    size: NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const dataSliceBuffer = device.createBuffer({
    size: 2 * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });
  const projectedSliceBuffer = device.createBuffer({
    size: 2 * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });

  const sortPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "sortProjections",
      module: device.createShaderModule({
        label: "Sort Gaussian Projections",
        code: `
${PROJECTED_GUASSIAN_DEF}
${COMMON_DEFS}  

const NUM_BITS_PER_BUCKET: u32 = 8;
const NUM_PASSES: u32 = 32 / NUM_BITS_PER_BUCKET;
const NUM_BUCKETS: u32 = 1 << NUM_BITS_PER_BUCKET;

@group(0) @binding(0) var<storage, read> input: array<ProjectedGaussian>;
@group(0) @binding(1) var<storage, read_write> output: array<ProjectedGaussian>;

/**
 * Stores the starting index of each key-bucket for the tile _after_ the
 * inspected tile. In other words, the first histogram will be set to the global
 * histogram plus the number of each key within the first tile.
 */
@group(0) @binding(2) var<storage, read_write> localHistograms: array<array<atomic<u32>, NUM_BUCKETS>>;

/**
 * The slice of the input array we are reading from.
 * THE LENGTH MUST BE A MULTIPLE OF tileLength
 */
@group(1) @binding(0) var<uniform> slice: Slice;
@group(1) @binding(1) var<storage, read> globalHistogram: array<u32, NUM_BUCKETS>;
@group(1) @binding(2) var<storage, read_write> nextTileIndex: atomic<u32>;
@group(1) @binding(3) var<uniform> passIndex: u32;

const blockSize: u32 = 32;
const itemsPerThreadPerTile: u32 = 16;
const tileLength = itemsPerThreadPerTile * blockSize;

const LOCAL_RESOLVED_FLAG: u32 = 1 << 30;
const GLOBAL_RESOLVED_FLAG: u32 = 1 << 31;
const VALUE_MASK: u32 = ~(LOCAL_RESOLVED_FLAG | GLOBAL_RESOLVED_FLAG);

var<workgroup> tileIndex: u32;

/**
 * Contains the inclusive prefix sum.
 */
var<workgroup> prefixSum: array<u32, NUM_BUCKETS>;

var<workgroup> scratchpad: array<u32, blockSize * itemsPerThreadPerTile>;

@compute @workgroup_size(blockSize)
fn sortProjections(
  @builtin(local_invocation_index) localIndex: u32,
) {
  if (localIndex == 0) {
    tileIndex = atomicAdd(&nextTileIndex, 1);
  }

  let tileStart = workgroupUniformLoad(&tileIndex) * tileLength;

  // this for loop will set the scratchpad to contain a bunch of
  // key-concat-index u32 values that we will then sort. While looping, we add
  // to the local histogram value to keep track of how many of each key we see.
  // We use the local histogram because it's already allocated ram.
  for (var i = localIndex; i < tileLength; i += blockSize) {
    let sortKey = input[slice.offset + tileStart + i].sortKey;
    let key = extractBits(
      select(0xFFFFFFFF, sortKey, sortKey != 0),
      passIndex * NUM_BITS_PER_BUCKET, 
      NUM_BITS_PER_BUCKET
    );

    atomicAdd(&localHistograms[tileIndex][key], 1);

    // combine the key with the index, we'll sort that combined value
    scratchpad[i] = insertBits(i, key, 32 - NUM_BITS_PER_BUCKET, NUM_BITS_PER_BUCKET);
  }

  workgroupBarrier();

  // Load up how much of each bucket there is
  for (var key: u32 = localIndex; key < NUM_BUCKETS; key += blockSize) {
    // this sum is currently how many values we have for that key
    var sum = atomicLoad(&localHistograms[tileIndex][key]);

    // will get turned into prefix sum later
    prefixSum[key] = sum;
    
    if (tileIndex == 0) {
      sum += GLOBAL_RESOLVED_FLAG + globalHistogram[key];
    } else {
      // save our current value, which allows other blocks to see it
      atomicStore(&localHistograms[tileIndex][key], LOCAL_RESOLVED_FLAG + sum);

      var loopCount: u32 = 0;

      var prevTile: u32 = tileIndex - 1;
      loop {
        let prevValue = atomicLoad(&localHistograms[prevTile][key]);

        if ((prevValue & GLOBAL_RESOLVED_FLAG) != 0) {
          // the prevValue comes with the flag bit, sort of convenient
          sum += prevValue;
          break;
        } else if ((prevValue & LOCAL_RESOLVED_FLAG) != 0) {
          sum += (prevValue & VALUE_MASK);

          // we know that prevTile is > 0 because the 0th tile will never write the LOCAL_RESOLVED_FLAG,
          // it will only ever write the GLOBAL_RESOLVED_FLAG or no flag at all
          prevTile--;
        } else {
          // loop back on this tile

          loopCount++;
          if (loopCount > 10000) {
            // prevents freezing the computer if something goes wrong
            // this case should never be hit
            break;
          }
        }
      }
    }

    // store our globally resolved value
    atomicStore(&localHistograms[tileIndex][key], sum);
  }

  workgroupBarrier();

  // as long as this assert is true, we know that looping backwards from NUM_BUCKETS will hit exactly 0
  const_assert (NUM_BUCKETS % blockSize == 0);

  // perform inclusive prefix sum
  for (var i: u32 = 1; i < NUM_BUCKETS; i *= 2) {
    // we start from the back so that we don't mess up 
    // later data with earlier writes
    for (var j: u32 = NUM_BUCKETS; j != 0; j -= blockSize) {
      let key = j - blockSize + localIndex;

      var earlierSum: u32 = 0;
      if (key >= i) {
        earlierSum = prefixSum[key - i];
      }
      workgroupBarrier();

      prefixSum[key] += earlierSum;
      workgroupBarrier();
    }
  }

  // at this point, prefixSum is the inclusive prefixSum
  // and the value in localHistograms is the global value for
  // how large each bucket is inclusive of this tile

  // copied from wikipedia pseudo-code
  // https://en.wikipedia.org/wiki/Bitonic_sorter
  for (var k: u32 = 2; k <= tileLength; k *= 2) {
    for (var j: u32 = k/2; j > 0; j /= 2) {
      for (var i = localIndex; i < tileLength; i += blockSize) {
        let neighbor = i ^ j;
        if (neighbor > i) {
          let mine = scratchpad[i];
          let theirs = scratchpad[neighbor];

          if (select((mine < theirs), (mine > theirs), (i & k) == 0)) {
            scratchpad[i] = theirs;
            scratchpad[neighbor] = mine;
          }
        }
      }

      workgroupBarrier();
    }
  }

  // now the scratchpad is sorted

  for (var i = localIndex; i < tileLength; i += blockSize) {
    let keyAndIndex: u32 = scratchpad[i];
    let key: u32 = extractBits(keyAndIndex, 32 - NUM_BITS_PER_BUCKET, NUM_BITS_PER_BUCKET);
    let index: u32 = extractBits(keyAndIndex, 0, 32 - NUM_BITS_PER_BUCKET);

    let bucketEnd = atomicLoad(&localHistograms[tileIndex][key]) & VALUE_MASK;

    output[
      slice.offset +
      bucketEnd - (prefixSum[key] - i)
    ] = input[
      slice.offset +
      tileStart +
      index
    ];
  }
}

  `,
      }),
    },
  });

  const bucketizePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "findBucketIndices",
      module: device.createShaderModule({
        label: "Shader to determine start indices of each bucket",
        code: `
${PROJECTED_GUASSIAN_DEF}
${COMMON_DEFS}

@group(0) @binding(0) var<storage> gaussians: array<ProjectedGaussian>;
@group(0) @binding(1) var<storage, read_write> buckets: array<u32>;
@group(0) @binding(2) var<uniform> slice: Slice;

override blockSize: u32 = 256;
override bitsForIndex: u32 = 16;

fn bucketOf(index: u32) -> u32 {
  return gaussians[slice.offset + index].sortKey >> (32 - bitsForIndex);
}

@compute @workgroup_size(blockSize)
fn findBucketIndices(
  @builtin(global_invocation_id) globalIndex: vec3u,
) {
  let index = globalIndex.x;
  if (index >= slice.length) {
    return;
  }

  let key = bucketOf(index);

  if (index == 0 || bucketOf(index - 1) < key) {
    buckets[2 * key] = index;
  }

  if (index + 1 == slice.length || key < bucketOf(index + 1)) {
    buckets[2 * key + 1] = index + 1;
  }
}

        
        `,
      }),
    },
  });

  const guassianPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "renderGaussians",
      module: device.createShaderModule({
        label: "Gaussian Splatting Shader",
        code: `
${PROJECTED_GUASSIAN_DEF}
${COMMON_DEFS}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<storage> renderables: array<ProjectedGaussian>;
@group(1) @binding(1) var<uniform> slice: Slice;

override chunkSize: u32 = 16;

@compute @workgroup_size(chunkSize, chunkSize)
fn renderGaussians(
  @builtin(workgroup_id) chunkPosition: vec3u,
  @builtin(local_invocation_id) offset: vec3u,
  @builtin(global_invocation_id) pixel: vec3u,
) {
  let coords = 2 * vec2f(pixel.xy) / vec2f(textureDimensions(outputTexture)) - 1;
  var color = vec4f(0, 0, 0, 0);

  let end = slice.offset + slice.length;
  for (var i = slice.offset; i < end; i++) {
    let in = renderables[i];
    let origin = in.origin;

    if (origin.z < 0.1) {
      continue;
    }

    var centered = coords - origin.xy;

    let Σ_inv = mat2x2f(
      in.Σ_inv.x, in.Σ_inv.y,
      in.Σ_inv.y, in.Σ_inv.z
    );

    let power = -.5 * dot(centered, Σ_inv * centered);
    if (power > 0) {
      continue;
    }

    let alpha = min(.99, exp(power) * in.color.w) * (1 - color.w);

    color += vec4f(alpha * in.color.xyz, alpha);
  }

  textureStore(outputTexture, pixel.xy, color);
}
    `,
      }),
    },
  });

  const guassianBindGroup = device.createBindGroup({
    layout: guassianPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: outputTexture.createView() }],
  });

  const cameraBuffer = device.createBuffer({
    size: 4 * 4 * NUM_BYTES_FLOAT32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let projectedGaussianBufferSize = 0;
  let projectedGaussianBuffers: undefined | Array<GPUBuffer>;
  let projectedDataBindGroups: undefined | Array<GPUBindGroup>;
  let binSizingDataBindGroups: undefined | Array<GPUBindGroup>;
  let histogramsBuffer: undefined | GPUBuffer;
  let sortPassDataBindGroups: undefined | Array<Array<GPUBindGroup>>;
  let bucketRangesBuffer: undefined | GPUBuffer;
  let bucketizeDataBindGroups: undefined | Array<GPUBindGroup>;
  let renderDataBindGroups: undefined | Array<GPUBindGroup>;

  // creates a shader that needs to draw 4 points, one for each corner of the screen
  const outputShader = device.createShaderModule({
    label: "Single Texture Shader",
    code: `
${COMMON_DEFS}

@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var screenTexture: texture_2d<f32>;
// @group(0) @binding(2) var<storage, read> debug_histogram: array<array<u32, 256>, 4>;
@group(0) @binding(2) var<storage, read> buckets: array<u32>;
@group(0) @binding(3) var<storage, read> localHistograms: array<array<u32, 256>>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) fragmentXY: vec2f,
}

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  const fragPoints = array(
    vec2f(0, 0),
    vec2f(1, 0),
    vec2f(0, 1),
    vec2f(1, 1),
  );

  let fragXY = fragPoints[index];

  return VertexOutput(
    // position
    vec4f(2 * fragXY - 1, 0, 1),
    // fragmentXY
    fragXY,
  );
}

@fragment
fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
  let dim = 16. / 512.;

  // let bucket = u32(floor(fragUV.y / dim)) * 32 + u32(floor(fragUV.x / dim));
  var tint = vec4f(0, 0, 0, 1);

  let bucket = u32(fragUV.y / dim) * 32 + u32(fragUV.x / dim);

  _ = localHistograms[0][0];
  // tint.x = f32(localHistograms[0][u32(fragUV.x * 256)] & 0x3FFFFFFF) / 300.;
  // tint.x = f32(localHistograms[0][bucket >> 8] & 0x3FFFFFFF) / 300.;

  let rangeStart = buckets[2 * bucket];
  let rangeEnd = buckets[2 * bucket + 1];
  tint.y = f32(rangeEnd - rangeStart) / 10.;
  // tint.y = f32(rangeStart) / 300.;
  // tint.z = f32(rangeEnd) / 300.;
  // tint.y = f32(range.start) / 1000.;
  // tint.y = f32(range.end) / 1024.;
  // tint.y = select(0., 1., buckets[bucket + 1].start == range.end);

  let color = textureSample(screenTexture, screenSampler, fragUV);
  
  return saturate(tint) * (1 - color.w) + color.w * color;
}
`,
  });

  const outputSampler = device.createSampler({
    label: "Basic Texture Sampler",
  });

  const renderTexturePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: outputShader,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: outputShader,
      entryPoint: "fragment_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-strip",
    },
  });

  const prefixSumBindGroup = device.createBindGroup({
    layout: prefixSumPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: histogramBuffer } }],
  });

  let renderTextureBindGroup: undefined | GPUBindGroup;
  // const renderTextureBindGroup = device.createBindGroup({
  //   layout: screenPipeline.getBindGroupLayout(0),
  //   entries: [
  //     { binding: 0, resource: outputSampler },
  //     {
  //       binding: 1,
  //       resource: outputTexture.createView(),
  //     },
  //     { binding: 2, resource: { buffer: histogramBuffer } },
  //   ],
  // });

  return render;

  function render(numSplats: number, cameraMatrix: Float32Array) {
    const numTiles = Math.ceil(numSplats / TILE_SIZE);
    const encoder = device.createCommandEncoder();

    if (numSplats !== projectedGaussianBufferSize) {
      projectedGaussianBuffers?.forEach((buffer) => {
        buffer.destroy();
      });
      histogramsBuffer?.destroy();
      bucketRangesBuffer?.destroy();

      writeToBuffer(
        device.queue,
        dataSliceBuffer,
        new Uint32Array([0, numSplats])
      );

      writeToBuffer(
        device.queue,
        projectedSliceBuffer,
        new Uint32Array([0, numTiles * TILE_SIZE])
      );

      projectedGaussianBuffers = ["", " (scratch-space)"].map((name) =>
        device.createBuffer({
          label: `Projected Gaussians Buffer${name} (size ${numSplats})`,
          size: numTiles * NUM_BYTES_TILE,
          usage: GPUBufferUsage.STORAGE,
        })
      );

      projectedDataBindGroups = [
        device.createBindGroup({
          label: "Specific Gaussian Data to Project onto Screen Space",
          layout: projectGaussiansPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: splatData } },
            { binding: 1, resource: { buffer: cameraBuffer } },
            { binding: 2, resource: { buffer: projectedGaussianBuffers[0] } },
            { binding: 3, resource: { buffer: dataSliceBuffer } },
          ],
        }),
      ];

      histogramsBuffer = device.createBuffer({
        label: `Local Histograms used in Sorting`,
        size: numTiles * NUM_BUCKETS * NUM_BYTES_UINT32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      binSizingDataBindGroups = [
        device.createBindGroup({
          label: "Bin Sizing Data",
          layout: binSizingPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffers[0] } },
            { binding: 1, resource: { buffer: dataSliceBuffer } },
            { binding: 2, resource: { buffer: histogramBuffer } },
            { binding: 3, resource: { buffer: nextTileIndexBuffer } },
          ],
        }),
      ];

      sortPassDataBindGroups = [0, 1, 0, 1].map((oddRound, i) => [
        device.createBindGroup({
          layout: sortPipeline.getBindGroupLayout(0),
          entries: [
            {
              binding: 0,
              resource: { buffer: projectedGaussianBuffers![oddRound] },
            },
            {
              binding: 1,
              resource: { buffer: projectedGaussianBuffers![1 - oddRound] },
            },
            { binding: 2, resource: { buffer: histogramsBuffer! } },
          ],
        }),
        device.createBindGroup({
          layout: sortPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: projectedSliceBuffer } },
            {
              binding: 1,
              resource: {
                buffer: histogramBuffer,
                offset: i * HISTOGRAM_SIZE,
                size: HISTOGRAM_SIZE,
              },
            },
            { binding: 2, resource: { buffer: nextTileIndexBuffer } },
            {
              binding: 3,
              resource: {
                buffer: constantsBuffers[constants.indexOf(i)],
              },
            },
          ],
        }),
      ]);

      bucketRangesBuffer = device.createBuffer({
        label: `An array holding bucket ranges`,
        size: chunkDims.y * chunkDims.x * 2 * NUM_BYTES_UINT32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      bucketizeDataBindGroups = [
        device.createBindGroup({
          label: "Bucketize Bind Group",
          layout: bucketizePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffers[0] } },
            { binding: 1, resource: { buffer: bucketRangesBuffer } },
            { binding: 2, resource: { buffer: dataSliceBuffer } },
          ],
        }),
      ];

      renderDataBindGroups = [
        guassianBindGroup,
        device.createBindGroup({
          layout: guassianPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffers[0] } },
            { binding: 1, resource: { buffer: dataSliceBuffer } },
          ],
        }),
      ];

      renderTextureBindGroup = device.createBindGroup({
        layout: renderTexturePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: outputSampler },
          {
            binding: 1,
            resource: outputTexture.createView(),
          },
          { binding: 2, resource: { buffer: bucketRangesBuffer } },
          { binding: 3, resource: { buffer: histogramsBuffer } },
        ],
      });
    }

    writeToBuffer(device.queue, cameraBuffer, cameraMatrix);

    const projectionPass = encoder.beginComputePass();
    projectionPass.setPipeline(projectGaussiansPipeline);
    projectedDataBindGroups!.forEach((group, i) => {
      projectionPass.setBindGroup(i, group);
    });
    projectionPass.dispatchWorkgroups(Math.ceil(numSplats / 256));
    projectionPass.end();

    // reset the histogram
    encoder.clearBuffer(histogramBuffer);
    encoder.clearBuffer(nextTileIndexBuffer);

    const binSizingPass = encoder.beginComputePass();
    binSizingPass.setPipeline(binSizingPipeline);
    binSizingDataBindGroups!.forEach((group, i) => {
      binSizingPass.setBindGroup(i, group);
    });
    // the number should be proportional to the number of gpu cores
    binSizingPass.dispatchWorkgroups(64);
    binSizingPass.end();

    encoder.clearBuffer(nextTileIndexBuffer);

    const prefixSumPass = encoder.beginComputePass();
    prefixSumPass.setPipeline(prefixSumPipeline);
    prefixSumPass.setBindGroup(0, prefixSumBindGroup);
    prefixSumPass.dispatchWorkgroups(4);
    prefixSumPass.end();

    sortPassDataBindGroups!.forEach((groups, i) => {
      encoder.clearBuffer(nextTileIndexBuffer);
      encoder.clearBuffer(histogramsBuffer!);

      const sortPass = encoder.beginComputePass();
      sortPass.setPipeline(sortPipeline);
      groups.forEach((group, i) => {
        sortPass.setBindGroup(i, group);
      });
      sortPass.dispatchWorkgroups(numTiles);
      sortPass.end();
    });

    encoder.clearBuffer(bucketRangesBuffer!);

    const bucketizePass = encoder.beginComputePass();
    bucketizePass.setPipeline(bucketizePipeline);
    bucketizeDataBindGroups!.forEach((group, i) => {
      bucketizePass.setBindGroup(i, group);
    });
    bucketizePass.dispatchWorkgroups(Math.ceil(numSplats / 256));
    bucketizePass.end();

    const guassianPass = encoder.beginComputePass();
    guassianPass.setPipeline(guassianPipeline);
    renderDataBindGroups!.forEach((group, i) => {
      guassianPass.setBindGroup(i, group);
    });
    guassianPass.dispatchWorkgroups(chunkDims.x, chunkDims.y);
    guassianPass.end();

    const outputPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    outputPass.setPipeline(renderTexturePipeline);
    outputPass.setBindGroup(0, renderTextureBindGroup!);
    outputPass.draw(4);
    outputPass.end();

    device.queue.submit([encoder.finish()]);
  }
}
