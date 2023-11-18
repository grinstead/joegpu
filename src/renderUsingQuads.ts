import { GPUCanvasDetails } from "./GPUCanvas.tsx";
import { QUAD_VERTICES } from "./gpu_utils.ts";
import { NUM_PROPERTIES_PLY, PLY_PROPERTY_INDEX } from "./ply.ts";
import { NUM_BYTES_FLOAT32 } from "./utils.ts";

export function renderUsingQuads(
  props: GPUCanvasDetails,
  splatData: GPUBuffer
) {
  const { canvas, context, device, format } = props;

  const vertices = new Float32Array(QUAD_VERTICES);
  const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(vertexBuffer, 0, vertices);

  const splatShader = device.createShaderModule({
    label: "Splat Render Shader",
    code: `

struct RenderParams {
  projectionMat: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> render_params: RenderParams;

struct GaussianSplat {
  @location(0) origin: vec3<f32>,
  @location(1) render_vertex: vec2<f32>,
  @location(2) color_sh0: vec3<f32>,
  @location(3) opacity: f32,
  @location(4) scales: vec3<f32>,
  @location(5) quarternion: vec4<f32>,
  @location(6) normal: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) Σ_prime_inv: vec3<f32>,
  @location(2) @interpolate(linear) quad_position: vec2<f32>,
}

const HARMONIC_COEFF0: f32 = 0.28209479177387814;

/*


let J = mat4x4(
  uniforms.focal_x / t.z, 0., -(uniforms.focal_x * t.x) / (t.z * t.z), 0.,
  0., uniforms.focal_y / t.z, -(uniforms.focal_y * t.y) / (t.z * t.z), 0.,
  0., 0., 0., 0.,
  0., 0., 0., 0.,
);
    (a, 0, b, 0)
J = (0, c, d, 0)
    (0, 0, 0, 0)
    (0, 0, 0, 0)

*/

fn invert_2x2(input: mat2x2<f32>) -> mat2x2<f32> {
  return (1 / determinant(input)) * mat2x2<f32>(
    input[1][1], -input[0][1],
    -input[1][0], input[0][0],
  );
}

// the opacity from the file is a sigmoid thing useful for training
fn normalize_opacity(in: f32) -> f32 {
  if (in >= 0) {
    return 1 / (1 + exp(-in));
  } else {
    let temp = exp(in);
    return temp / (1 + temp);
  }
}

@vertex
fn vertex_main(in: GaussianSplat) -> VertexOutput {
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

  let camera_space_origin = render_params.projectionMat * vec4<f32>(in.origin, 1.0);
  let z = camera_space_origin.z;

  let JW = mat4x4<f32>(
    1 / z, 0, 0, 0,
    0, 1 / z, 0, 0,
    -1 / z / z, -1 / z / z, 0, 0,
    0, 0, 0, 0,
  ) * render_params.projectionMat; 

  // x in camera space -> x coordinate in screen space
  // x as is for now, but z^-1 so derivative is -z^-2

  var Σ_prime_full = JW * Σ * transpose(JW);

  let det = Σ_prime_full[0][0] * Σ_prime_full[1][1] - Σ_prime_full[0][1] * Σ_prime_full[1][0];
  let Σ_prime_inv = invert_2x2(
    mat2x2<f32>(
      Σ_prime_full[0][0], Σ_prime_full[0][1],
      Σ_prime_full[1][0], Σ_prime_full[1][1],
    )
  );

  let mid = 0.5 * (Σ_prime_full[0][0] + Σ_prime_full[1][1]);
  let sqrtThing = sqrt(max(0.1, mid * mid - det));
  let radius_camera_space = 3 * sqrt(sqrtThing) / 256.;
  // let radius_camera_space = .01;

  var out: VertexOutput;
  out.position = vec4<f32>(
    camera_space_origin.xy + in.render_vertex * radius_camera_space * z, 
    z * z / 100, 
    z
  );
  out.color = vec4<f32>(in.color_sh0 * HARMONIC_COEFF0 + .5, normalize_opacity(in.opacity));
  out.Σ_prime_inv = vec3<f32>(Σ_prime_inv[0][0], Σ_prime_inv[0][1], Σ_prime_inv[1][1]);
  out.quad_position = radius_camera_space * in.render_vertex;
  return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
  if (in.position.z < .005) {
    discard;
  }

  let Σ_prime_inv = mat2x2<f32>(
    in.Σ_prime_inv.x, in.Σ_prime_inv.y, 
    in.Σ_prime_inv.y, in.Σ_prime_inv.z,
  );

  let power = -.5 * dot(in.quad_position, Σ_prime_inv * in.quad_position);
  if (power > 0) {
    discard;
  }

  let alpha: f32 = min(exp(power) * in.color.w, .99);

  if (alpha < .5) {
    discard;
  }

  return vec4<f32>(in.color.xyz, alpha);
}
`,
  });

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const cameraMatrix = new Float32Array(16);

  const cameraBuffer = device.createBuffer({
    size: cameraMatrix.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const splatRenderPipeline = device.createRenderPipeline({
    layout: "auto",
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth32float",
    },
    vertex: {
      module: splatShader,
      entryPoint: "vertex_main",
      buffers: [
        {
          arrayStride: NUM_PROPERTIES_PLY * NUM_BYTES_FLOAT32,
          stepMode: "instance",
          attributes: [
            {
              // position
              shaderLocation: 0,
              offset: PLY_PROPERTY_INDEX.position * NUM_BYTES_FLOAT32,
              format: "float32x3",
            },
            {
              // color_coeff0
              shaderLocation: 2,
              offset: PLY_PROPERTY_INDEX.color * NUM_BYTES_FLOAT32,
              format: "float32x3",
            },
            {
              // opacity
              shaderLocation: 3,
              offset: PLY_PROPERTY_INDEX.opacity * NUM_BYTES_FLOAT32,
              format: "float32",
            },
            {
              // scales
              shaderLocation: 4,
              offset: PLY_PROPERTY_INDEX.logScales * NUM_BYTES_FLOAT32,
              format: "float32x3",
            },
            {
              // quarternion
              shaderLocation: 5,
              offset: PLY_PROPERTY_INDEX.quarternion * NUM_BYTES_FLOAT32,
              format: "float32x4",
            },
            {
              // normals
              shaderLocation: 6,
              offset: PLY_PROPERTY_INDEX.normals * NUM_BYTES_FLOAT32,
              format: "float32x3",
            },
          ],
        },
        {
          arrayStride: 2 * NUM_BYTES_FLOAT32,
          attributes: [
            {
              format: "float32x2",
              offset: 0,
              shaderLocation: 1,
            },
          ],
        },
      ],
    },
    fragment: {
      module: splatShader,
      entryPoint: "fragment_main",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-strip",
    },
  });

  const cameraBufferBindGroup = device.createBindGroup({
    layout: splatRenderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: cameraBuffer,
        },
      },
    ],
  });

  function render(numSplats: number, cameraMatrix: Float32Array) {
    device.queue.writeBuffer(
      cameraBuffer,
      0,
      cameraMatrix.buffer,
      cameraMatrix.byteOffset,
      cameraMatrix.byteLength
    );

    const encoder = device.createCommandEncoder();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 33 / 256, g: 33 / 256, b: 33 / 256, a: 0 },
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(splatRenderPipeline);
    pass.setBindGroup(0, cameraBufferBindGroup);
    pass.setVertexBuffer(0, splatData);
    pass.setVertexBuffer(1, vertexBuffer);
    pass.draw(vertices.length / 2, numSplats, 0, 0);

    pass.end();

    device.queue.submit([encoder.finish()]);
  }

  return render;
}
