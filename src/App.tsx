import { Show, createSignal } from "solid-js";
import "./App.css";
import { GPUCanvas, GPUCanvasProps, createGPUCanvas } from "./GPUCanvas";
import { Result } from "./utils";

export function App() {
  const [result, setResult] = createSignal<Result<GPUCanvasProps>>();

  createGPUCanvas().then(setResult, (error) =>
    setResult({ success: false, error })
  );

  return (
    <Show when={result()} fallback={<div>Loading...</div>}>
      {(r) => {
        const item = r();
        return item.success ? (
          <GPUCanvas {...item.value} />
        ) : (
          String(item.error)
        );
      }}
    </Show>
  );
}
