import * as glm from 'gl-matrix';

//================================//
interface Camera
{
    position: Float32Array;
    forward: Float32Array;
    up: Float32Array;
    right: Float32Array;
    worldUp: Float32Array;

    fovY: number;
    aspect: number;
    near: number;
    far: number;

    yaw: number;
    pitch: number;

    moveSpeed: number;
    rotateSpeed: number;

    modelMatrix: Float32Array;
    viewMatrix: Float32Array;
    projectionMatrix: Float32Array;
}

//================================//
export function createCamera(aspect: number): Camera
{
    const camera: Camera = {
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

        modelMatrix: mat4Identity(),
        viewMatrix: mat4Identity(),
        projectionMatrix: mat4Perspective(Math.PI / 4, aspect, 0.1, 1000),
    };

    updateCameraVectors(camera);
    return camera;
}

//================================//
export function setCameraPosition(camera: Camera, x: number, y: number, z: number): void
{
    camera.position[0] = x;
    camera.position[1] = y;
    camera.position[2] = z;
    updateViewMatrix(camera);
}

//================================//
export function setCameraFovY(camera: Camera, fovY: number): void
{
    camera.fovY = fovY;
    updateProjectionMatrix(camera);
}

//================================//
export function setCameraAspect(camera: Camera, aspect: number): void
{
    camera.aspect = aspect;
    updateProjectionMatrix(camera);
}

//================================//
export function setCameraNearFar(camera: Camera, near: number, far: number): void
{
    camera.near = near;
    camera.far = far;
    updateProjectionMatrix(camera);
}

//================================//
export function setCameraRotation(camera: Camera, yaw: number, pitch: number): void
{
    camera.yaw = yaw;
    camera.pitch = pitch;
    
    // Clamp pitch to avoid flipping
    const maxPitch = Math.PI / 2 - 0.01;
    camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, camera.pitch));
    
    updateCameraVectors(camera);
}

//================================//
export function moveCameraBy(camera: Camera, dx: number, dy: number, dz: number): void
{
    camera.position[0] += dx;
    camera.position[1] += dy;
    camera.position[2] += dz;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraForward(camera: Camera, amount: number): void
{
    camera.position[0] += camera.forward[0] * amount;
    camera.position[1] += camera.forward[1] * amount;
    camera.position[2] += camera.forward[2] * amount;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraRight(camera: Camera, amount: number): void
{
    camera.position[0] += camera.right[0] * amount;
    camera.position[1] += camera.right[1] * amount;
    camera.position[2] += camera.right[2] * amount;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraUp(camera: Camera, amount: number): void
{
    camera.position[0] += camera.up[0] * amount;
    camera.position[1] += camera.up[1] * amount;
    camera.position[2] += camera.up[2] * amount;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraWorldUp(camera: Camera, amount: number): void
{
    camera.position[1] += amount;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraForwardXZ(camera: Camera, amount: number): void
{
    const forwardXZ = new Float32Array([camera.forward[0], 0, camera.forward[2]]);
    vec3Normalize(forwardXZ);
    camera.position[0] += forwardXZ[0] * amount;
    camera.position[2] += forwardXZ[2] * amount;
    updateViewMatrix(camera);
}

//================================//
export function moveCameraLocal(camera: Camera, forward: number, right: number, up: number): void
{
    camera.position[0] += camera.forward[0] * forward + camera.right[0] * right + camera.up[0] * up;
    camera.position[1] += camera.forward[1] * forward + camera.right[1] * right + camera.up[1] * up;
    camera.position[2] += camera.forward[2] * forward + camera.right[2] * right + camera.up[2] * up;
    updateViewMatrix(camera);
}

//================================//
export function rotateCameraBy(camera: Camera, deltaYaw: number, deltaPitch: number): void
{
    camera.yaw += deltaYaw;
    camera.pitch += deltaPitch;
    
    // Clamp pitch to avoid flipping
    const maxPitch = Math.PI / 2 - 0.01;
    camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, camera.pitch));
    
    // Normalize yaw to [-PI, PI]
    while (camera.yaw > Math.PI) camera.yaw -= 2 * Math.PI;
    while (camera.yaw < -Math.PI) camera.yaw += 2 * Math.PI;
    
    updateCameraVectors(camera);
}

//================================//
export function rotateCameraByMouse(camera: Camera, deltaX: number, deltaY: number): void
{
    rotateCameraBy(camera, deltaX * camera.rotateSpeed, deltaY * camera.rotateSpeed);
}

//================================//
export function lookInDirection(camera: Camera, dirX: number, dirY: number, dirZ: number): void
{
    const dir = new Float32Array([dirX, dirY, dirZ]);
    vec3Normalize(dir);
    
    camera.yaw = Math.atan2(dir[2], dir[0]);
    camera.pitch = Math.asin(dir[1]);
    
    const maxPitch = Math.PI / 2 - 0.01;
    camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, camera.pitch));
    
    updateCameraVectors(camera);
}

//================================//
export function lookAtPoint(camera: Camera, targetX: number, targetY: number, targetZ: number): void
{
    const dirX = targetX - camera.position[0];
    const dirY = targetY - camera.position[1];
    const dirZ = targetZ - camera.position[2];
    lookInDirection(camera, dirX, dirY, dirZ);
}

//================================//
export function getCameraForward(camera: Camera): Float32Array
{
    return new Float32Array(camera.forward);
}

//================================//
export function getCameraRight(camera: Camera): Float32Array
{
    return new Float32Array(camera.right);
}

//================================//
export function getCameraUp(camera: Camera): Float32Array
{
    return new Float32Array(camera.up);
}

//================================//
export function getCameraPosition(camera: Camera): Float32Array
{
    return new Float32Array(camera.position);
}

//================================//
export function getCameraTarget(camera: Camera): Float32Array
{
    // Returns the point the camera is looking at (position + forward)
    return new Float32Array([
        camera.position[0] + camera.forward[0],
        camera.position[1] + camera.forward[1],
        camera.position[2] + camera.forward[2],
    ]);
}

//================================//
function updateCameraVectors(camera: Camera): void
{
    // Calculate forward vector from yaw and pitch
    camera.forward[0] = Math.cos(camera.pitch) * Math.cos(camera.yaw);
    camera.forward[1] = Math.sin(camera.pitch);
    camera.forward[2] = Math.cos(camera.pitch) * Math.sin(camera.yaw);
    vec3Normalize(camera.forward);
    
    // Recalculate right and up vectors
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
function updateViewMatrix(camera: Camera): void
{
    const target = new Float32Array([
        camera.position[0] + camera.forward[0],
        camera.position[1] + camera.forward[1],
        camera.position[2] + camera.forward[2],
    ]);
    camera.viewMatrix = mat4LookAt(camera.position, target, camera.up);
}

//================================//
function updateProjectionMatrix(camera: Camera): void
{
    camera.projectionMatrix = mat4Perspective(camera.fovY, camera.aspect, camera.near, camera.far);
}

//================================//
function mat4Identity(): Float32Array
{
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

//================================//
function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Float32Array
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
function mat4LookAt(eye: Float32Array, target: Float32Array, up: Float32Array): Float32Array
{
    const zAxis = new Float32Array([
        eye[0] - target[0],
        eye[1] - target[1],
        eye[2] - target[2],
    ]);
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
function vec3Normalize(v: Float32Array): void
{
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 0.00001) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

//================================//
function vec3Cross(a: Float32Array, b: Float32Array): Float32Array
{
    return new Float32Array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]);
}

//================================//
function vec3Dot(a: Float32Array, b: Float32Array): number
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//================================//
export function computePixelToRayMatrix(camera: Camera): Float32Array
{
    const tanHalfFov = Math.tan(camera.fovY / 2.0);
    const scaleX = camera.aspect * tanHalfFov;
    const scaleY = tanHalfFov;
    
    // WebGPU uses column-major layout
    return new Float32Array([
        // Column 0: right * scaleX
        camera.right[0] * scaleX,
        camera.right[1] * scaleX,
        camera.right[2] * scaleX,
        0,
        // Column 1: up * scaleY
        camera.up[0] * scaleY,
        camera.up[1] * scaleY,
        camera.up[2] * scaleY,
        0,
        // Column 2: forward (z = 1 in NDC input)
        camera.forward[0],
        camera.forward[1],
        camera.forward[2],
        0,
        // Column 3: unused (w = 0 for directions)
        0,
        0,
        0,
        1,
    ]);
}

//================================//
export function validatePixelToRayMatrix(camera: Camera): boolean
{
    const M = computePixelToRayMatrix(camera);
    
    // Test: center pixel (ndc = 0, 0) with z=1 should give forward direction
    // M * vec4(0, 0, 1, 0) = column 2 = forward
    const centerRay = new Float32Array([M[8], M[9], M[10]]);
    vec3Normalize(centerRay);
    
    const forward = new Float32Array(camera.forward);
    vec3Normalize(forward);
    
    const dot = vec3Dot(centerRay, forward);
    const isValid = Math.abs(dot - 1.0) < 1e-5;
    
    if (!isValid) {
        console.error("PixelToRayMatrix validation failed: center ray does not match forward direction");
        console.log("Center ray:", centerRay);
        console.log("Forward:", forward);
        console.log("Dot product:", dot);
    }
    
    return isValid;
}

//================================//
// Knowing camera parameters, and NDC coordinates which can be screen coordinates, 
// return direction ray in world space
export function cameraPointToRay(camera: Camera, ndcX: number, ndcY: number): Float32Array
{
    const M = computePixelToRayMatrix(camera);
    
    // NDC to ray direction in world space
    const rayDir = new Float32Array([
        M[0] * ndcX + M[4] * ndcY + M[8] * 1.0,
        M[1] * ndcX + M[5] * ndcY + M[9] * 1.0,
        M[2] * ndcX + M[6] * ndcY + M[10] * 1.0,
    ]);
    vec3Normalize(rayDir);
    return rayDir;
}

//================================//
export function rayIntersectsSphere(rayOrigin: Float32Array, rayDir: Float32Array, sphereCenter: glm.vec3, sphereRadius: number): number
{
    const L = new Float32Array([
        sphereCenter[0] - rayOrigin[0],
        sphereCenter[1] - rayOrigin[1],
        sphereCenter[2] - rayOrigin[2],
    ]);
    const tca = vec3Dot(L, rayDir);
    if (tca < 0) return -1;

    const d2 = vec3Dot(L, L) - tca * tca;
    const radius2 = sphereRadius * sphereRadius;
    if (d2 > radius2) return -1;
    

    // return distance
    const thc = Math.sqrt(radius2 - d2);
    const t0 = tca - thc;
    
    // Inside the sphere?
    if (t0 < 0) return -1;
    
    return t0;
}