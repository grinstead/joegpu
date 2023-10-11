import { Result } from "./utils";

export type GPUCanvasDetails = {
  context: GPUCanvasContext;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  format: GPUTextureFormat;
};

export type GPUCanvasProps = {
  details: GPUCanvasDetails;
  render: (details: GPUCanvasDetails) => void;
};

export function GPUCanvas(props: GPUCanvasProps) {
  props.render(props.details);
  return props.details.canvas;
}

export async function createGPUCanvas(): Promise<Result<GPUCanvasDetails>> {
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

  const format = gpu.getPreferredCanvasFormat();

  context.configure({ device, format });

  return {
    success: true,
    value: { context, device, canvas, format },
  };
}
