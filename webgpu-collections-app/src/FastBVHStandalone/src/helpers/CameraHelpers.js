//================================//
export function createCamera(aspect)
{
    const camera = {
        position: new Float32Array([0, 0, 4]),
        forward: new Float32Array([0, 0, 1]),
        up: new Float32Array([0, 1, 0]),
        right: new Float32Array([1, 0, 0]),
        worldUp: new Float32Array([0, 1, 0]),

        fovY: Math.PI / 4,
        aspect: aspect,
        near: 0.1,
        far: 1000,

        yaw: Math.PI / 2,
        pitch: 0,

        moveSpeed: 0.01,
        rotateSpeed: 0.5,

        viewMatrix: mat4Identity(),
        projectionMatrix: mat4Perspective(Math.PI / 4, aspect, 0.1, 1000),

        dirty: true,
    };

    updateCameraVectors(camera);
    return camera;
}

//================================//
export function setCameraPosition(camera, x, y, z)
{
    camera.position[0] = x;
    camera.position[1] = y;
    camera.position[2] = z;
    updateViewMatrix(camera);
}

//================================//
export function setCameraAspect(camera, aspect)
{
    camera.aspect = aspect;
    updateProjectionMatrix(camera);
}

//================================//
export function setCameraNearFar(camera, near, far)
{
    camera.near = near;
    camera.far = far;
    updateProjectionMatrix(camera);
}

//================================//
export function moveCameraLocal(camera, forward, right, up)
{
    camera.position[0] += camera.forward[0] * forward + camera.right[0] * right + camera.up[0] * up;
    camera.position[1] += camera.forward[1] * forward + camera.right[1] * right + camera.up[1] * up;
    camera.position[2] += camera.forward[2] * forward + camera.right[2] * right + camera.up[2] * up;
    updateViewMatrix(camera);
}

//================================//
export function rotateCameraBy(camera, deltaYaw, deltaPitch)
{
    camera.yaw += deltaYaw;
    camera.pitch += deltaPitch;

    const maxPitch = Math.PI / 2 - 0.01;
    camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, camera.pitch));

    while (camera.yaw > Math.PI) camera.yaw -= 2 * Math.PI;
    while (camera.yaw < -Math.PI) camera.yaw += 2 * Math.PI;

    updateCameraVectors(camera);
}

//================================//
export function rotateCameraByMouse(camera, deltaX, deltaY)
{
    rotateCameraBy(camera, deltaX * camera.rotateSpeed, deltaY * camera.rotateSpeed);
}

//================================//
export function computePixelToRayMatrix(camera)
{
    const tanHalfFov = Math.tan(camera.fovY / 2.0);
    const scaleX = camera.aspect * tanHalfFov;
    const scaleY = tanHalfFov;

    return new Float32Array([
        camera.right[0] * scaleX,
        camera.right[1] * scaleX,
        camera.right[2] * scaleX,
        0,
        camera.up[0] * scaleY,
        camera.up[1] * scaleY,
        camera.up[2] * scaleY,
        0,
        camera.forward[0],
        camera.forward[1],
        camera.forward[2],
        0,
        0, 0, 0, 1,
    ]);
}

//================================//
export function cameraPointToRay(camera, ndcX, ndcY)
{
    const M = computePixelToRayMatrix(camera);

    const rx = M[0] * ndcX + M[4] * ndcY + M[8];
    const ry = M[1] * ndcX + M[5] * ndcY + M[9];
    const rz = M[2] * ndcX + M[6] * ndcY + M[10];
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz);

    const rayDir = new Float32Array([rx / len, ry / len, rz / len]);
    return {
        origin: camera.position,
        direction: rayDir,
        invDir: new Float32Array([1.0 / rayDir[0], 1.0 / rayDir[1], 1.0 / rayDir[2]]),
    };
}

//================================//
function updateCameraVectors(camera)
{
    camera.forward[0] = Math.cos(camera.pitch) * Math.cos(camera.yaw);
    camera.forward[1] = Math.sin(camera.pitch);
    camera.forward[2] = Math.cos(camera.pitch) * Math.sin(camera.yaw);
    vec3Normalize(camera.forward);

    const right = vec3Cross(camera.forward, camera.worldUp);
    vec3Normalize(right);
    camera.right[0] = right[0];
    camera.right[1] = right[1];
    camera.right[2] = right[2];

    const up = vec3Cross(camera.right, camera.forward);
    vec3Normalize(up);
    camera.up[0] = up[0];
    camera.up[1] = up[1];
    camera.up[2] = up[2];

    updateViewMatrix(camera);
}

//================================//
function updateViewMatrix(camera)
{
    const target = new Float32Array([
        camera.position[0] + camera.forward[0],
        camera.position[1] + camera.forward[1],
        camera.position[2] + camera.forward[2],
    ]);
    camera.viewMatrix = mat4LookAt(camera.position, target, camera.up);
    camera.dirty = true;
}

//================================//
function updateProjectionMatrix(camera)
{
    camera.projectionMatrix = mat4Perspective(camera.fovY, camera.aspect, camera.near, camera.far);
    camera.dirty = true;
}

//================================//
function mat4Identity()
{
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

//================================//
function mat4Perspective(fovy, aspect, near, far)
{
    const f = 1.0 / Math.tan(fovy * 0.5);
    const nf = 1.0 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far * nf, -1,
        0, 0, near * far * nf, 0,
    ]);
}

//================================//
function mat4LookAt(eye, target, up)
{
    const zAxis = new Float32Array([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    vec3Normalize(zAxis);

    const xAxis = vec3Cross(up, zAxis);
    vec3Normalize(xAxis);

    const yAxis = vec3Cross(zAxis, xAxis);

    return new Float32Array([
        xAxis[0], yAxis[0], zAxis[0], 0,
        xAxis[1], yAxis[1], zAxis[1], 0,
        xAxis[2], yAxis[2], zAxis[2], 0,
        -vec3Dot(xAxis, eye), -vec3Dot(yAxis, eye), -vec3Dot(zAxis, eye), 1,
    ]);
}

//================================//
function vec3Normalize(v)
{
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 0.00001) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

//================================//
function vec3Cross(a, b)
{
    return new Float32Array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]);
}

//================================//
function vec3Dot(a, b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
