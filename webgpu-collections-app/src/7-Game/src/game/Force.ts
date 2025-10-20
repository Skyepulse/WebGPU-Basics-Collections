/*
 * Force.ts
 *
 * Represents a force applied between two bodies.
 *
 */

import * as glm from 'gl-matrix';
import type RigidBox from "./RigidBox";
import type { ContactRender } from './Manifold';
class Force
{
    public bodyA: RigidBox;
    public bodyB: RigidBox;

    public static readonly MAX_ROWS: number = 4;

    public J: glm.vec3[] = []; // VECTOR3
    public H: glm.mat3[] = []; // MATRIX3
    public C: number[] = [];
    public fmin: number[] = [];
    public fmax: number[] = [];
    public stiffness: number[] = [];
    public fracture: number[] = [];
    public penalty: number[] = [];
    public lambda: number[] = [];

    //=============== PUBLIC =================//
    constructor(bodyA: RigidBox, bodyB: RigidBox)
    {
        this.bodyA = bodyA;
        this.bodyB = bodyB;

        for (let i = 0; i < Force.MAX_ROWS; ++i)
        {
            this.J.push(glm.vec3.fromValues(0, 0, 0));

            const m = glm.mat3.create();
            this.H.push(m);

            this.C.push(0);
            this.fmin.push(-Infinity);
            this.fmax.push(Infinity);
            this.stiffness.push(Infinity);
            this.fracture.push(Infinity);
            this.penalty.push(0);
            this.lambda.push(0);
        }
    }

    //================================//
    public disable(): void
    {
        for (let i = 0; i < Force.MAX_ROWS; ++i)
        {
            this.stiffness[i] = 0;
            this.penalty[i] = 0;
            this.lambda[i] = 0;
        }
    }

    //============= VIRTUAL ===================//
    public initialize(): boolean
    {
        console.warn("This method should not be called directly.");
        return true;
    }

    //================================//
    public computeConstraints(alpha: number): void
    {
        console.warn("This method should not be called directly.");
        alpha = alpha;
        return;
    }

    //================================//
    public computeDerivatives(body: RigidBox): void
    {
        console.warn("This method should not be called directly.");
        body = body;
        return;
    }

    //================================//
    public getRows(): number
    {
        console.warn("This method should not be called directly.");
        return 0;
    }

    //================================//
    public getContactRenders(): ContactRender[]
    {
        console.warn("This method should not be called directly.");
        return [];
    }
}

export default Force;