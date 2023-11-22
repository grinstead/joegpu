import { Show, createSignal } from "solid-js";
import "./App.css";
import { GPUCanvas, GPUCanvasDetails, createGPUCanvas } from "./GPUCanvas";
import { NUM_BYTES_FLOAT32, Result } from "./utils";
import exampleDataUrl from "./assets/example.ply?url";
import { findString } from "./searchBytes.ts";
import { NUM_PROPERTIES_PLY } from "./ply.ts";
import { renderUsingQuads } from "./renderUsingQuads.ts";
import { MutatingMatrix } from "./matrix.ts";
import { renderUsingSort } from "./renderUsingSort.ts";

type Point = { x: number; y: number };

export function App() {
  const [result, setResult] = createSignal<Result<GPUCanvasDetails>>();

  createGPUCanvas().then(setResult, (error) =>
    setResult({ success: false, error })
  );

  return (
    <>
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
    </>
  );
}

function projectedAngle(x: number) {
  return Math.atan2(Math.sqrt(1 - x * x), x);
}

function clamp(value: number, min: number, max: number) {
  return value < min ? min : value < max ? value : max;
}

async function renderAppCanvas(props: GPUCanvasDetails) {
  const fileBytes = await (await fetch(exampleDataUrl)).arrayBuffer();

  const bodyIndex =
    findString(new Uint8Array(fileBytes), "end_header\n") +
    "end_header\n".length;

  const { device, canvas } = props;

  let numSplats =
    (fileBytes.byteLength - bodyIndex) / NUM_BYTES_FLOAT32 / NUM_PROPERTIES_PLY;
  numSplats = Math.min(numSplats, 1200);

  const USE_OLD_RENDER = false;

  const splatDataBuffer = device.createBuffer({
    label: "Splat Data",
    size: numSplats * NUM_BYTES_FLOAT32 * NUM_PROPERTIES_PLY,
    usage: USE_OLD_RENDER
      ? GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
      : GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  device.queue.writeBuffer(
    splatDataBuffer,
    0,
    fileBytes,
    bodyIndex,
    numSplats * NUM_BYTES_FLOAT32 * NUM_PROPERTIES_PLY
  );

  const CANVAS_RADIUS = 256;

  let rotation = { x: -0.279, y: -0.371, z: 0 };
  let zoom = { x: 1, y: 1, z: 1 };
  let translation = { x: -1.383, y: -0.387, z: 2.001 };
  let renderUntil = 0;

  let mouseStatus:
    | undefined
    | {
        meta: "shift" | "alt" | undefined;
        startPoint: Point;
        lastPoint: Point;
        translation: typeof translation;
        lastTimeMs: number;
      };

  canvas.addEventListener("mousemove", (event) => {
    // only respond to events that have a mouse down
    if ((event.buttons & 1) === 0) {
      return;
    }

    if (!renderUntil) {
      renderUntil = Date.now();
      requestAnimationFrame(render);
    }

    const { offsetX, offsetY } = event;
    let x = clamp(offsetX / CANVAS_RADIUS, 0, 2) - 1;
    let y = 1 - clamp(offsetY / CANVAS_RADIUS, 0, 2);

    const point = { x, y };

    const meta = event.shiftKey ? "shift" : event.altKey ? "alt" : undefined;

    const now = Date.now();
    if (
      mouseStatus &&
      (now > mouseStatus.lastTimeMs + 100 || meta !== mouseStatus.meta)
    ) {
      mouseStatus = undefined;
    }

    if (!mouseStatus) {
      mouseStatus = {
        meta,
        startPoint: point,
        lastPoint: point,
        lastTimeMs: now,
        translation: { ...translation },
      };
      return;
    }

    if (meta === "shift") {
      const { startPoint } = mouseStatus;
      const startMag = Math.sqrt(startPoint.x ** 2 + startPoint.y ** 2);
      const dot = startPoint.x * x + startPoint.y * y;

      const zScale = dot / startMag / startMag;

      translation.z = mouseStatus.translation.z / zScale;
      rotation.z +=
        Math.atan2(y, x) -
        Math.atan2(mouseStatus.lastPoint.y, mouseStatus.lastPoint.x);
    } else if (meta === "alt") {
      rotation.y += projectedAngle(x) - projectedAngle(mouseStatus.lastPoint.x);
      rotation.x -= projectedAngle(y) - projectedAngle(mouseStatus.lastPoint.y);
    } else {
      translation.x += x - mouseStatus.lastPoint.x;
      translation.y += y - mouseStatus.lastPoint.y;
    }

    mouseStatus.lastPoint = point;
    mouseStatus.lastTimeMs = now;

    (window as any).DEBUG_INFO = { translation, zoom, rotation };
  });

  const cameraMatrix = new Float32Array(16);

  const renderImpl = (USE_OLD_RENDER ? renderUsingQuads : renderUsingSort)(
    props,
    splatDataBuffer
  );

  function render() {
    new MutatingMatrix(cameraMatrix)
      .reset()
      .scale(1, -1, 1)
      .rotateAroundX(rotation.x)
      .rotateAroundY(rotation.y)
      .rotateAroundZ(rotation.z)
      .scale(zoom.x, zoom.y, zoom.z)
      .translate(translation.x, translation.y, translation.z)
      .transpose();

    renderImpl(numSplats, cameraMatrix);

    if (renderUntil > Date.now()) {
      requestAnimationFrame(render);
    } else {
      renderUntil = 0;
    }
  }

  render();
}
