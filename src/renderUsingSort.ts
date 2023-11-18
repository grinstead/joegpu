import { GPUCanvasDetails } from "./GPUCanvas.tsx";

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
  const guassianShader = device.createShaderModule({
    label: "Gaussian Splatting Shader",
    code: `
@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;

override chunkSize: u32 = 16;

@compute @workgroup_size(chunkSize, chunkSize)
fn renderGuassians(
  @builtin(workgroup_id) chunkPosition: vec3u,
  @builtin(local_invocation_id) offset: vec3u,
  @builtin(global_invocation_id) pixel: vec3u,
) {
  textureStore(outputTexture, pixel.xy, vec4f(vec2f(offset.xy) / f32(chunkSize), 0, 1));
}
`,
  });

  const guassianPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: guassianShader,
      entryPoint: "renderGuassians",
    },
  });

  const guassianBindGroup = device.createBindGroup({
    layout: guassianPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: outputTexture.createView() }],
  });

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
  let fromTexture = textureSample(screenTexture, screenSampler, fragUV);

  // currently mixing in other colors just to confirm things are rendering
  return (fromTexture + vec4f(0, fragUV, 1)) * .5;
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

    const guassianPass = encoder.beginComputePass();
    guassianPass.setPipeline(guassianPipeline);
    guassianPass.setBindGroup(0, guassianBindGroup);
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
