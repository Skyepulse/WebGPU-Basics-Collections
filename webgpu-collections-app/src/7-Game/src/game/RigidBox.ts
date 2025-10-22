/*
 * RigidBox.ts
 *
 * Responsible for representing a 2D rigid body with a box shape.
 *
 */

import * as glm from 'gl-matrix';
import Force from './Force';

//================================//
class RigidBox 
{
    private width: number;
    private height: number;

    private mass: number;
    private density: number;

    private friction: number;

    private position: glm.vec3;
    private velocity: glm.vec3;
    public prevVelocity: glm.vec3;

    private color: Uint8Array;
    private staticBody: boolean;

    private moment: number = 0;
    private radius: number = 0;

    public lastPosition: glm.vec3 = glm.vec3.fromValues(0,0,0);
    public inertial: glm.vec3 = glm.vec3.fromValues(0, 0, 0);

    public id = -1;

    public forces: Force[] = [];

    //=============== PUBLIC =================//
    constructor(scale: glm.vec2, color: Uint8Array, density: number, friction: number, position: glm.vec3, velocity: glm.vec3)
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
        this.moment         = this.mass * glm.vec2.dot(scale, scale) / 12;
        this.radius         = Math.sqrt(glm.vec2.dot(scale, scale)) * 0.5;

        this.color          = color;
    }

    //================================//
    public getScale(): glm.vec2 { return glm.vec2.fromValues(this.width, this.height); }
    public getDensity(): number { return this.density; }
    public getMass(): number { return this.mass; }
    public getPosition(): glm.vec3 { return this.position; }
    public getPos2(): glm.vec2 { return glm.vec2.fromValues(this.position[0], this.position[1]); }
    public getColor(): Uint8Array { return this.color; }

    public getVelocity(): glm.vec3 { return this.velocity; }
    public getPrevVelocity(): glm.vec3 { return this.prevVelocity; }
    public getFriction(): number { return this.friction; }
    public isStatic(): boolean { return this.staticBody; }
    public getMoment(): number { return this.moment; }
    public getRadius(): number { return this.radius; }

    //================================//
    public setVelocity(velocity: glm.vec3): void { if (!this.staticBody) this.velocity = velocity; }

    //================================//
    public getRotationMatrix(): glm.mat2
    {
        const c = Math.cos(this.position[2]);
        const s = Math.sin(this.position[2]);

        const mat = glm.mat2.fromValues( c, s,
                                         -s,  c) ;
        return mat;
    }

    //================================//
    public setPosition(position: glm.vec3): void { if (!this.staticBody) this.position = position; }

    //================================//
    public setColor(color: Uint8Array): void { this.color = color; }

    //================================//
    public isConstrainedTo(body: RigidBox): boolean
    {
        for (let i = 0; i < this.forces.length; ++i)
        {
            const f = this.forces[i];

            if (f.bodyA === this && f.bodyB === body) return true;
            if (f.bodyB === this && f.bodyA === body) return true;
        }

        return false;
    }
}

export default RigidBox;