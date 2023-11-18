import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { NUM_BYTES_FLOAT32 } from "./utils.ts";

const CHUNK_SIZE = 16;

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

  // This shader renders the guassian splats to a texture
  const gaussianShader = device.createShaderModule({
    label: "Gaussian Splatting Shader",
    code: `

struct GaussianSplat {
  origin: vec3f,
  normal: vec3f,
  color_sh0: vec3f,
  color_rest: array<vec3f, 15>,
  opacity: f32,
  scales: vec3f,
  quarternion: vec4f,
}

struct ProjectedGaussian {
  origin: vec3f,
  Σ_inv: vec3f,
  color: vec4f,
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<storage> gaussians: array<GaussianSplat>;
@group(1) @binding(1) var<uniform> camera: mat4x4f; 
@group(1) @binding(2) var<storage, read_write> projectedGaussians: array<ProjectedGaussian>;
@group(2) @binding(0) var<storage> renderables: array<ProjectedGaussian>;

override chunkSize: u32 = 16;
override projectionChunkSize: u32 = 16;

const HARMONIC_COEFF0: f32 = 0.28209479177387814;

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

@compute @workgroup_size(projectionChunkSize)
fn projectGaussians(
  @builtin(global_invocation_id) index: vec3u,
) {
  if (index.x > arrayLength(&gaussians)) {
    return;
  }

  let in = gaussians[index.x];

  let camera_space_origin = camera * vec4<f32>(in.origin, 1.0);
  let z = camera_space_origin.z;

  // quarternion to matrix formula taken from
  // https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
  let q0 = in.quarternion[0];
  let q1 = in.quarternion[1];
  let q2 = in.quarternion[2];
  let q3 = in.quarternion[3];

  // R (rotation) and S (scales) matrices from Gaussian Splat Paper
  // technically these are the transposed versions because the gpu is col-major order
  let R = mat4x4<f32>(
    2*(q0*q0 + q1*q1) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2), 0,
    2*(q1*q2 + q0*q3), 2*(q0*q0 + q2*q2) - 1, 2*(q2*q3 - q0*q1), 0,
    2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0*q0 + q3*q3) - 1, 0,
    0, 0, 0, 1
  );
  let SR_T = mat4x4<f32>(
    exp(in.scales[0]), 0, 0, 0,
    0, exp(in.scales[1]), 0, 0,
    0, 0, exp(in.scales[2]), 0,
    0, 0, 0, 1,
  ) * R;

  // Σ is from Gaussian Splat paper (section 4, eq. 6)
  let Σ = transpose(SR_T) * SR_T;

  let JW = mat4x4<f32>(
    1 / z, 0, 0, 0,
    0, 1 / z, 0, 0,
    -1 / z / z, -1 / z / z, 0, 0,
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
      in.color_sh0 * HARMONIC_COEFF0 + .5,
      normalize_opacity(in.opacity),
    ),
  );
}

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

    let centered = coords - origin.xy;

    if (length(centered) < .01) {
      let splatColor = vec4<f32>(
        in.color.xyz,
        1,
      ); 
      color = splatColor;
    }

    // let power = -.5 * dot(centered, Σ_prime_inv * centered);
    // if (power < 0) {
    //   let alpha = min(.99, exp(power) * normalize_opacity(in.opacity));
    //   let splatColor = vec4<f32>(
    //     in.color_sh0 * HARMONIC_COEFF0 + .5,
    //     1,
    //   );

    //   color = (1 - alpha) * color + alpha * splatColor;
    //   // color = vec4f(0, alpha, 0, 1);
    // }
  }

  textureStore(outputTexture, pixel.xy, color);
}
`,
  });

  const projectGaussiansPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: gaussianShader,
      entryPoint: "projectGaussians",
    },
  });

  const guassianPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: gaussianShader,
      entryPoint: "renderGaussians",
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

    device.queue.writeBuffer(
      cameraBuffer,
      0,
      cameraMatrix.buffer,
      cameraMatrix.byteOffset,
      cameraMatrix.byteLength
    );

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
          layout: projectGaussiansPipeline.getBindGroupLayout(0),
          entries: [],
        }),
        device.createBindGroup({
          label: "Specific Gaussian Data to Render",
          layout: projectGaussiansPipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: splatData } },
            { binding: 1, resource: { buffer: cameraBuffer } },
            { binding: 2, resource: { buffer: projectedGaussianBuffer } },
          ],
        }),
      ];

      renderDataBindGroups = [
        guassianBindGroup,
        device.createBindGroup({
          layout: guassianPipeline.getBindGroupLayout(1),
          entries: [],
        }),
        device.createBindGroup({
          layout: guassianPipeline.getBindGroupLayout(2),
          entries: [
            { binding: 0, resource: { buffer: projectedGaussianBuffer } },
          ],
        }),
      ];
    }

    const projectionPass = encoder.beginComputePass();
    projectionPass.setPipeline(projectGaussiansPipeline);
    projectedDataBindGroups!.forEach((group, i) => {
      projectionPass.setBindGroup(i, group);
    });
    projectionPass.dispatchWorkgroups(Math.ceil(numSplats / CHUNK_SIZE));
    projectionPass.end();

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

  return render;
}
