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
  const { device, context } = props;

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

  pass.end();

  device.queue.submit([encoder.finish()]);
}
