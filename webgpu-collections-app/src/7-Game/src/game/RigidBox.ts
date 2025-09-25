/*
 * RigidBox.ts
 *
 * Responsible for representing a 2D rigid body with a box shape.
 *
 */

import { dot2 } from "@src/helpers/MathUtils";

//================================//
class RigidBox 
{
    private width: number;
    private height: number;

    private mass: number;
    private density: number;

    private friction: number;

    private position: Float32Array;
    private velocity: Float32Array;
    private prevVelocity: Float32Array;

    private color: Uint8Array;
    private staticBody: boolean;

    private moment: number = 0;
    private radius: number = 0;
    
    public id = -1;

    //=============== PUBLIC =================//
    constructor(scale: Float32Array, color: Uint8Array, density: number, friction: number, position: Float32Array, velocity: Float32Array)
    {
        this.width          = scale[0];
        this.height         = scale[1];

        this.density        = density;
        this.mass           = this.width * this.height * this.density;
        this.staticBody     = (this.mass === 0);
        this.friction       = friction;

        this.position       = position;
        this.velocity       = velocity;
        this.prevVelocity   = velocity;
        this.moment         = this.mass * dot2(scale, scale) / 12;
        this.radius         = Math.sqrt(dot2(scale, scale)) * 0.5;

        this.color          = color;
    }

    //================================//
    public getScale(): Float32Array { return new Float32Array([this.width, this.height]); }
    public getDensity(): number { return this.density; }
    public getMass(): number { return this.mass; }
    public getPosition(): Float32Array { return this.position; }
    public getColor(): Uint8Array { return this.color; }

    //================================//
    public setPosition(position: Float32Array): void { if (!this.staticBody) this.position = position; }
    public setColor(color: Uint8Array): void { this.color = color; }
}

export default RigidBox;