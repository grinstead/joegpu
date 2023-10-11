import { Component, JSX, Show, createSignal } from "solid-js";
import "./App.css";

export function App() {
  const [result, setResult] = createSignal<Result<Component>>();

  createGPUCanvas().then(setResult, (error) =>
    setResult({ success: false, error })
  );

  return (
    <Show when={result()} fallback={<div>Loading...</div>}>
      {(r) => {
        const Item = r();
        return Item.success ? <Item.value /> : String(Item.error);
      }}
    </Show>
  );
}

type Result<T, Error = unknown> =
  | {
      success: true;
      value: T;
    }
  | {
      success: false;
      error: Error;
    };

async function createGPUCanvas(): Promise<Result<() => JSX.Element>> {
  const { gpu } = navigator as Partial<NavigatorGPU>;

  if (!gpu) {
    return {
      success: false,
      error: "This browser is not supported, consider Chrome",
    };
  }

  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    return {
      success: false,
      error: "This hardware is not supported",
    };
  }

  const device = await adapter.requestDevice();

  const canvas = (<canvas width={512} height={512} />) as HTMLCanvasElement;

  const context = canvas.getContext("webgpu");

  if (!context) {
    return {
      success: false,
      error: "Failed to initialize canvas",
    };
  }

  context.configure({
    device,
    format: gpu.getPreferredCanvasFormat(),
  });

  return {
    success: true,
    value: () => <GPUCanvas device={device} canvas={canvas} />,
  };
}

type GPUCanvasProps = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
};

function GPUCanvas(props: GPUCanvasProps) {
  return props.canvas;
}
