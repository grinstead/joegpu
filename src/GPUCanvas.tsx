import { Result } from "./utils";

export type GPUCanvasProps = {
  context: GPUCanvasContext;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
};

export function GPUCanvas(props: GPUCanvasProps) {
  const { device, context, canvas } = props;

  const encoder = device.createCommandEncoder();

  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.end();

  device.queue.submit([encoder.finish()]);

  return canvas;
}

export async function createGPUCanvas(): Promise<Result<GPUCanvasProps>> {
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
    value: { context, device, canvas },
  };
}
