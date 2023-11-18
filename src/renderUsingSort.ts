import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { QUAD_VERTICES } from "./gpu_utils.ts";

export function renderUsingSort(props: GPUCanvasDetails, splatData: GPUBuffer) {
  const { canvas, context, device, format } = props;

  // creates a shader that needs to draw 4 points, one for each corner of the screen
  const screenShader = device.createShaderModule({
    label: "Single Texture Shader",
    code: `

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

  var output: VertexOutput;
  output.position = vec4f(2 * fragXY - 1, 0, 1);
  output.fragmentXY = fragXY;
  return output;
}

@fragment
fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
  return vec4f(0, fragUV, 1);
}   
`,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: screenShader,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: screenShader,
      entryPoint: "fragment_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-strip",
    },
  });

  const encoder = device.createCommandEncoder();

  return render;

  function render(numSplats: number, cameraMatrix: Float32Array) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.draw(4);
    pass.end();

    device.queue.submit([encoder.finish()]);
  }

  return render;
}
