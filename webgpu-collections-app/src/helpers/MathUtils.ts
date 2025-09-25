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
export function randomPosInRect(x: number, y: number, width: number, height: number): Float32Array
{
  return new Float32Array([rand(x, x + width), rand(y, y + height)]);
};

//================================//
export function randomPosInRectRot(x: number, y: number, width: number, height: number): Float32Array
{
  return new Float32Array([rand(x, x + width), rand(y, y + height), rand(0, Math.PI * 2)]);
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