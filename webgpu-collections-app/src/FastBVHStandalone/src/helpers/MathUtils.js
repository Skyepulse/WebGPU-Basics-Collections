//================================//
export function rotationMatrix3(angleX, angleY, angleZ)
{
    const cx = Math.cos(angleX);
    const sx = Math.sin(angleX);
    const cy = Math.cos(angleY);
    const sy = Math.sin(angleY);
    const cz = Math.cos(angleZ);
    const sz = Math.sin(angleZ);

    // Returns column-major Float32Array (mat3)
    return new Float32Array([
        cy * cz,                     -cy * sz,                    sy,
        sx * sy * cz + cx * sz,     -sx * sy * sz + cx * cz,   -sx * cy,
        -cx * sy * cz + sx * sz,     cx * sy * sz + sx * cz,    cx * cy
    ]);
}

//================================//
export function computeNormal(v0, v1, v2)
{
    const ex = v1[0] - v0[0], ey = v1[1] - v0[1], ez = v1[2] - v0[2];
    const fx = v2[0] - v0[0], fy = v2[1] - v0[1], fz = v2[2] - v0[2];
    const nx = ey * fz - ez * fy;
    const ny = ez * fx - ex * fz;
    const nz = ex * fy - ey * fx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-10) return new Float32Array([0, 1, 0]);
    return new Float32Array([nx / len, ny / len, nz / len]);
}

//================================//
export function radsToDegrees(rads)
{
    return rads * (180 / Math.PI);
}

//================================//
export function degreesToRads(degrees)
{
    return degrees * (Math.PI / 180);
}
