import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { writeToBuffer } from "./gpu_utils.ts";
import { NUM_BYTES_FLOAT32, NUM_BYTES_UINT32 } from "./utils.ts";

const CHUNK_SIZE = 16;

const SLICE_DEF = `
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

export function renderUsingSort(props: GPUCanvasDetails, splatData: GPUBuffer) {
  const { canvas, context, device, format } = props;

  const chunkDims = {
    x: Math.ceil(canvas.width / CHUNK_SIZE),
    y: Math.ceil(canvas.height / CHUNK_SIZE),
  };

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
${SLICE_DEF}

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

override blockSize: u32 = 16;
@compute @workgroup_size(blockSize)
fn projectGaussians(
  @builtin(global_invocation_id) index: vec3u,
) {
  if (index.x >= slice.length) {
    return;
  }

  let in = gaussians[index.x + slice.offset];

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

  projectedGaussians[index.x] = ProjectedGaussian(
    screenSpace,
    min(chunkId.y, 31) * 256 + min(chunkId.x, 31),
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
${SLICE_DEF}

const NUM_BITS_PER_BUCKET: u32 = 8;
const NUM_PASSES: u32 = 32 / NUM_BITS_PER_BUCKET;
const NUM_BUCKETS: u32 = 1 << NUM_BITS_PER_BUCKET;

@group(0) @binding(0) var<storage> gaussians: array<ProjectedGaussian>;
@group(0) @binding(1) var<uniform> slice: Slice;

// these should always be empty
@group(0) @binding(2) var<storage, read_write> globalHistogram: array<array<atomic<u32>, NUM_BUCKETS>, NUM_PASSES>;
@group(0) @binding(3) var<storage, read_write> nextTileStart: atomic<u32>;

override blockSize: u32 = 128;
override itemsPerThreadPerTile: u32 = 16;

var<workgroup> workgroupStart: u32;
var<workgroup> localHistogram: array<array<atomic<u32>, NUM_BUCKETS>, NUM_PASSES>;

// currently do nothing
@compute @workgroup_size(blockSize)
fn computeBinSizes(
  @builtin(local_invocation_index) localIndex: u32,
) {  
  let tileSize = itemsPerThreadPerTile * blockSize;

  // loop through "tiles"
  loop {
    if (localIndex == 0) {
      workgroupStart = atomicAdd(&nextTileStart, tileSize);
    }

    let tileStart = workgroupUniformLoad(&workgroupStart);
    if (tileStart >= slice.length) {
      break;
    }

    let end = slice.offset + min(slice.length, tileStart + tileSize);

    for (
      var i: u32 = slice.offset + tileStart + localIndex;
      i < end;
      i += blockSize
    ) {
      let key = gaussians[i].sortKey;
      
      for (var round: u32 = 0; round < NUM_PASSES; round++) {
        let subkey = extractBits(
          key,
          round * NUM_BITS_PER_BUCKET,
          NUM_BITS_PER_BUCKET
        );
        atomicAdd(&localHistogram[round][subkey], 1);
      }
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
    size:
      4 /* number of passes */ *
      256 /* number of buckets per pass */ *
      NUM_BYTES_UINT32,
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

  const sortPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "sortProjections",
      module: device.createShaderModule({
        label: "Sort Gaussian Projections",
        code: `
${PROJECTED_GUASSIAN_DEF}
${SLICE_DEF}  

const NUM_BITS_PER_BUCKET: u32 = 8;
const NUM_PASSES: u32 = 32 / NUM_BITS_PER_BUCKET;
const NUM_BUCKETS: u32 = 1 << NUM_BITS_PER_BUCKET;

@group(0) @binding(0) var<storage, read> input: array<ProjectedGaussian>;
@group(0) @binding(1) var<storage, read_write> output: array<ProjectedGaussian>;
@group(1) @binding(0) var<uniform> slice: Slice;
@group(1) @binding(1) var<storage, read> globalHistogram: array<array<u32, NUM_BUCKETS>, NUM_PASSES>;
@group(1) @binding(2) var<storage, read_write> nextTileStart: atomic<u32>;

const passIndex = 0;

override blockSize: u32 = 32;
override itemsPerThreadPerTile: u32 = 16;

var<workgroup> tileStart: u32;
var<workgroup> localHistogram: array<atomic<u32>, NUM_BUCKETS>;
var<workgroup> scratchpad: array<u32, blockSize * itemsPerThreadPerTile>;

@compute @workgroup_size(blockSize)
fn sortProjections(
  @builtin(local_invocation_index) localIndex: u32,
) {
  let targetTileSize = itemsPerThreadPerTile * blockSize;

  if (localIndex == 0) {
    tileStart = atomicAdd(&nextTileStart, targetTileSize);
  }

  let tileLength: u32 = min(
    targetTileSize,
    slice.length - workgroupUniformLoad(&tileStart)
  );

  for (var i = localIndex; i < tileLength; i += blockSize) {
    let key = extractBits(
      input[slice.offset + tileStart + i].sortKey,
      passIndex * NUM_BITS_PER_BUCKET, 
      NUM_BITS_PER_BUCKET
    );

    atomicAdd(&localHistogram[key], 1);

    // combine the key with the index, we'll sort that combined value
    scratchpad[i] = insertBits(i, key, 32 - NUM_BITS_PER_BUCKET, NUM_BITS_PER_BUCKET);
  }

  workgroupBarrier();

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

  for (var i = localIndex; i < tileLength; i += blockSize) {
    output[slice.offset + tileStart + i] =
      input[
        slice.offset +
        tileStart +
        extractBits(
          scratchpad[i],
          0,
          32 - NUM_BITS_PER_BUCKET
        )
      ];
  }

  _ = globalHistogram[0][0];
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
${SLICE_DEF}

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
  var color = vec4f(.1, .1, .1, 0);

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

    let alpha = min(.99, exp(power) * in.color.w);

    color = (1 - alpha) * color + alpha * in.color;
    // color = vec4f(0, alpha, 0, 1);
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
  let sortPassDataBindGroups: undefined | Array<GPUBindGroup>;
  let renderDataBindGroups: undefined | Array<GPUBindGroup>;

  // creates a shader that needs to draw 4 points, one for each corner of the screen
  const outputShader = device.createShaderModule({
    label: "Single Texture Shader",
    code: `
@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var screenTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> debug_histogram: array<array<u32, 256>, 4>;

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

  if (fragUV.x < dim) {
    tint.x = f32(debug_histogram[1][u32(fragUV.y / dim)]) / 1000.;
  }

  if (fragUV.y < dim) {
    tint.y = f32(debug_histogram[0][u32(fragUV.x / dim)]) / 1000.;
  }

  let color = textureSample(screenTexture, screenSampler, fragUV);
  
  return saturate(tint) * (1 - color.w) + color.w * color;
}
`,
  });

  const outputSampler = device.createSampler({
    label: "Basic Texture Sampler",
  });

  const screenPipeline = device.createRenderPipeline({
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

  const renderTextureBindGroup = device.createBindGroup({
    layout: screenPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outputSampler },
      {
        binding: 1,
        resource: outputTexture.createView(),
      },
      { binding: 2, resource: { buffer: histogramBuffer } },
    ],
  });

  return render;

  function render(numSplats: number, cameraMatrix: Float32Array) {
    const encoder = device.createCommandEncoder();

    if (numSplats !== projectedGaussianBufferSize) {
      projectedGaussianBuffers?.forEach((buffer) => {
        buffer.destroy();
      });

      projectedGaussianBuffers = ["", " (scratch-space)"].map((name) =>
        device.createBuffer({
          label: `Projected Gaussians Buffer${name} (size ${numSplats})`,
          size: numSplats * 12 * NUM_BYTES_FLOAT32,
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

      sortPassDataBindGroups = [
        device.createBindGroup({
          layout: sortPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffers[0] } },
            { binding: 1, resource: { buffer: projectedGaussianBuffers[1] } },
          ],
        }),
        device.createBindGroup({
          layout: sortPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: dataSliceBuffer } },
            { binding: 1, resource: { buffer: histogramBuffer } },
            { binding: 2, resource: { buffer: nextTileIndexBuffer } },
          ],
        }),
      ];

      renderDataBindGroups = [
        guassianBindGroup,
        device.createBindGroup({
          layout: guassianPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffers[1] } },
            { binding: 1, resource: { buffer: dataSliceBuffer } },
          ],
        }),
      ];
    }

    writeToBuffer(device.queue, cameraBuffer, cameraMatrix);

    const projectionPass = encoder.beginComputePass();
    projectionPass.setPipeline(projectGaussiansPipeline);
    projectedDataBindGroups!.forEach((group, i) => {
      projectionPass.setBindGroup(i, group);
    });
    projectionPass.dispatchWorkgroups(Math.ceil(numSplats / CHUNK_SIZE));
    projectionPass.end();

    // reset the histogram
    encoder.clearBuffer(histogramBuffer);
    encoder.clearBuffer(nextTileIndexBuffer);

    writeToBuffer(
      device.queue,
      dataSliceBuffer,
      new Uint32Array([0, numSplats])
    );

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

    const sortPass0 = encoder.beginComputePass();
    sortPass0.setPipeline(sortPipeline);
    sortPassDataBindGroups!.forEach((group, i) => {
      sortPass0.setBindGroup(i, group);
    });
    sortPass0.dispatchWorkgroups(Math.ceil(numSplats / 512));
    sortPass0.end();

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

    outputPass.setPipeline(screenPipeline);
    outputPass.setBindGroup(0, renderTextureBindGroup);
    outputPass.draw(4);
    outputPass.end();

    device.queue.submit([encoder.finish()]);
  }
}
