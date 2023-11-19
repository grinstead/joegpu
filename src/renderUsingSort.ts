import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { NUM_BYTES_FLOAT32, NUM_BYTES_UINT32 } from "./utils.ts";

const CHUNK_SIZE = 16;

const PROJECTED_GUASSIAN_DEF = `
struct ProjectedGaussian {
  origin: vec3f,
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

const HARMONIC_COEFF0: f32 = 0.28209479177387814;

@group(0) @binding(0) var<storage> gaussians: array<GaussianSplat>;
@group(0) @binding(1) var<uniform> camera: mat4x4f; 
@group(0) @binding(2) var<storage, read_write> projectedGaussians: array<ProjectedGaussian>;

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

override blockSize: u32 = 16;
@compute @workgroup_size(blockSize)
fn projectGaussians(
  @builtin(global_invocation_id) index: vec3u,
) {
  if (index.x > arrayLength(&gaussians)) {
    return;
  }

  let in = gaussians[index.x];

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

  projectedGaussians[index.x] = ProjectedGaussian(
    vec3f(camera_space_origin.xy / z, z),
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

const NUM_PASSES = 4;
const NUM_BUCKETS = 256;

@group(0) @binding(0) var<storage> gaussians: array<ProjectedGaussian>;
@group(0) @binding(1) var<storage, read_write> histogramSizes: array<array<u32, NUM_BUCKETS>, NUM_PASSES>;

override chunkSize: u32 = 128;
override itemsPerThreadPerTile: u32 = 16;

// currently do nothing
@compute @workgroup_size(chunkSize)
fn computeBinSizes(

) {
  _ = gaussians[0];
  _ = histogramSizes[0][0];
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

  const guassianPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      entryPoint: "renderGaussians",
      module: device.createShaderModule({
        label: "Gaussian Splatting Shader",
        code: `
${PROJECTED_GUASSIAN_DEF}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<storage> renderables: array<ProjectedGaussian>;

override chunkSize: u32 = 16;

@compute @workgroup_size(chunkSize, chunkSize)
fn renderGaussians(
  @builtin(workgroup_id) chunkPosition: vec3u,
  @builtin(local_invocation_id) offset: vec3u,
  @builtin(global_invocation_id) pixel: vec3u,
) {
  let numGaussians = arrayLength(&renderables);
  let coords = 2 * vec2f(pixel.xy) / vec2f(textureDimensions(outputTexture)) - 1;
  var color = vec4f(.1, .1, .1, 0);

  for (var i: u32 = 0; i < numGaussians; i++) {
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
  let projectedGaussianBuffer: undefined | GPUBuffer;
  let projectedDataBindGroups: undefined | Array<GPUBindGroup>;
  let binSizingDataBindGroups: undefined | Array<GPUBindGroup>;
  let renderDataBindGroups: undefined | Array<GPUBindGroup>;

  // creates a shader that needs to draw 4 points, one for each corner of the screen
  const outputShader = device.createShaderModule({
    label: "Single Texture Shader",
    code: `
@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var screenTexture: texture_2d<f32>;

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
  return textureSample(screenTexture, screenSampler, fragUV);

  // currently mixing in other colors just to confirm things are rendering
  // return (fromTexture + vec4f(0, fragUV, 1)) * .5;
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

  const renderTextureBindGroup = device.createBindGroup({
    layout: screenPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outputSampler },
      {
        binding: 1,
        resource: outputTexture.createView(),
      },
    ],
  });

  return render;

  function render(numSplats: number, cameraMatrix: Float32Array) {
    const encoder = device.createCommandEncoder();

    if (numSplats !== projectedGaussianBufferSize) {
      if (projectedGaussianBuffer) {
        projectedGaussianBuffer.destroy();
      }

      projectedGaussianBuffer = device.createBuffer({
        label: `Projected Gaussians (${numSplats})`,
        size: numSplats * 10 * NUM_BYTES_FLOAT32,
        usage: GPUBufferUsage.STORAGE,
      });

      projectedDataBindGroups = [
        device.createBindGroup({
          label: "Specific Gaussian Data to Project onto Screen Space",
          layout: projectGaussiansPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: splatData } },
            { binding: 1, resource: { buffer: cameraBuffer } },
            { binding: 2, resource: { buffer: projectedGaussianBuffer } },
          ],
        }),
      ];

      binSizingDataBindGroups = [
        device.createBindGroup({
          label: "Bin Sizing Data",
          layout: binSizingPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffer } },
            { binding: 1, resource: { buffer: histogramBuffer } },
          ],
        }),
      ];

      renderDataBindGroups = [
        guassianBindGroup,
        device.createBindGroup({
          layout: guassianPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffer } },
          ],
        }),
      ];
    }

    device.queue.writeBuffer(
      cameraBuffer,
      0,
      cameraMatrix.buffer,
      cameraMatrix.byteOffset,
      cameraMatrix.byteLength
    );

    const projectionPass = encoder.beginComputePass();
    projectionPass.setPipeline(projectGaussiansPipeline);
    projectedDataBindGroups!.forEach((group, i) => {
      projectionPass.setBindGroup(i, group);
    });
    projectionPass.dispatchWorkgroups(Math.ceil(numSplats / CHUNK_SIZE));
    projectionPass.end();

    // reset the histogram
    encoder.clearBuffer(histogramBuffer);

    const binSizingPass = encoder.beginComputePass();
    binSizingPass.setPipeline(binSizingPipeline);
    binSizingDataBindGroups!.forEach((group, i) => {
      binSizingPass.setBindGroup(i, group);
    });
    // the number should be proportional to the number of gpu cores
    binSizingPass.dispatchWorkgroups(64);
    binSizingPass.end();

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
