import { Show, createSignal } from "solid-js";
import "./App.css";
import { GPUCanvas, GPUCanvasDetails, createGPUCanvas } from "./GPUCanvas";
import { Result } from "./utils";

export function App() {
  const [result, setResult] = createSignal<Result<GPUCanvasDetails>>();

  createGPUCanvas().then(setResult, (error) =>
    setResult({ success: false, error })
  );

  return (
    <Show when={result()} fallback={<div>Loading...</div>}>
      {(r) => {
        const item = r();
        return item.success ? (
          <GPUCanvas details={item.value} render={renderAppCanvas} />
        ) : (
          String(item.error)
        );
      }}
    </Show>
  );
}

function renderAppCanvas(props: GPUCanvasDetails) {
  const { device, context, format } = props;

  const encoder = device.createCommandEncoder();

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
        storeOp: "store",
      },
    ],
  });

  const GRID_SIZE = 32;
  const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
  const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  // prettier-ignore
  const vertices = new Float32Array([
    -0.8, -0.8, // Triangle 1 (Blue)
      0.8, -0.8,
      0.8,  0.8,

    -0.8, -0.8, // Triangle 2 (Red)
      0.8,  0.8,
    -0.8,  0.8,
  ]);

  const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(vertexBuffer, 0, vertices);

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8,
    attributes: [
      {
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
      },
    ],
  };

  const shader = device.createShaderModule({
    label: "Cell shader",
    code: `
@group(0) @binding(0) var<uniform> grid: vec2f;

@vertex
fn vertexMain(@location(0) pos: vec2f, @builtin(instance_index) instance: u32) -> @builtin(position) vec4f {
  let i = f32(instance);
  let cell = vec2f(i % grid.x, floor(i / grid.x));
  let cellOffset = cell / grid * 2;
  return vec4f((pos + 1) / grid - 1 + cellOffset, 0, 1);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(1, 0, 0, 1);
}
`,
  });

  const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shader,
      entryPoint: "fragmentMain",
      targets: [
        {
          format,
        },
      ],
    },
  });

  const bindGroup = device.createBindGroup({
    label: "Cell renderer bind group",
    layout: cellPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer },
      },
    ],
  });

  pass.setPipeline(cellPipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setBindGroup(0, bindGroup);
  pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

  pass.end();

  device.queue.submit([encoder.finish()]);
}
