import * as glm from 'gl-matrix';

//================================//
export function rotationMatrix(angle: number): glm.mat2
{
  const c = Math.cos(angle);
  const s = Math.sin(angle);

  return glm.mat2.fromValues( c, s, 
                              -s, c  );
}

//================================//
export function scaleByValue(m: glm.mat3, v: number): glm.mat3
{
  const out = glm.mat3.create();
  for (let i = 0; i < 9; ++i) out[i] = m[i] * v;
  return out;
}

//================================//
export function outerMult(a: glm.vec3, b: glm.vec3): glm.mat3
{
  const out = glm.mat3.create();
  out[0] = a[0] * b[0]; out[1] = a[0] * b[1]; out[2] = a[0] * b[2];
  out[3] = a[1] * b[0]; out[4] = a[1] * b[1]; out[5] = a[1] * b[2];
  out[6] = a[2] * b[0]; out[7] = a[2] * b[1]; out[8] = a[2] * b[2];
  return out;
}

//================================//
export function solveLDLT(A: glm.mat3, b: glm.vec3): glm.vec3
{
  // Decompose A into LDL^T
  let D1 = A[0];
  let L21 = A[3] / A[0];
  let L31 = A[6] / A[0];
  let D2 = A[4] - L21 * L21 * D1;
  let L32 = (A[7] - L31 * L21 * D1) / D2;
  let D3 = A[8] - (L31 * L31 * D1 + L32 * L32 * D2);

  // Ly = b
  let y1 = b[0];
  let y2 = b[1] - L21 * y1;
  let y3 = b[2] - L31 * y1 - L32 * y2;

  // Dz = y
  let z1 = y1 / D1;
  let z2 = y2 / D2;
  let z3 = y3 / D3;

  // L^Tx = z
  const x: glm.vec3 = glm.vec3.fromValues(0, 0, 0);
  x[2] = z3;
  x[1] = z2 - L32 * x[2];
  x[0] = z1 - L21 * x[1] - L31 * x[2];

  return x;
}

//================================//
export function rand(min: number = 0, max: number = 1)
{
  if (min === undefined) {
    min = 0;
    max = 1;
  } else if (max === undefined) {
    max = min;
    min = 0;
  }
  return min + Math.random() * (max - min);
};

//================================//
export function randomPosInRect(x: number, y: number, width: number, height: number): glm.vec2
{
  return glm.vec2.fromValues(rand(x, x + width), rand(y, y + height));
};

//================================//
export function randomPosInRectRot(x: number, y: number, width: number, height: number): glm.vec3
{
  return glm.vec3.fromValues(rand(x, x + width), rand(y, y + height), rand(0, Math.PI * 2));
};

//================================//
export function randomColorUint8(): Uint8Array
{
  const r = Math.floor(rand(0, 256));
  const g = Math.floor(rand(0, 256));
  const b = Math.floor(rand(0, 256));
  const a = 255;
  return new Uint8Array([r, g, b, a]);
}

//================================//
export function dot2(a: Float32Array, b: Float32Array): number
{
  return a[0] * b[0] + a[1] * b[1];
}

//================================//
export function dot3(a: Float32Array, b: Float32Array): number
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//================================//
export function cross2(a: glm.vec2, b: glm.vec2): number
{
  return a[0] * b[1] - a[1] * b[0];
}