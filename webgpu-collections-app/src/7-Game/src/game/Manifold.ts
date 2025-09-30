/*
 * Manifold.ts
 *
 * Represents a manifold constraint between two bodies. This mean two points of contact,
 * each with its own normal and friction coefficient. Each contact has two constraint rows.
 * One for the normal force and one for the friction (tangential) force.
 *
 * Based on Chris Gile's avbd2d C++ implementation
 */

// Box vertex and edge numbering:
//
//        ^ y
//        |
//        e1
//   v2 ------ v1
//    |        |
// e2 |        | e4  --> x
//    |        |
//   v3 ------ v4
//        e3

import Force from "./Force";
import type RigidBox from "./RigidBox";
import { rotationMatrix, cross2 } from "@src/helpers/MathUtils";
import * as glm from 'gl-matrix';

const COLLISION_MARGIN: number = 0.0005;
const STICK_THRESHOLD: number = 0.01;

//================================//
enum Edges
{
    NO_EDGE = 0,
    EDGE1 = 1,
    EDGE2 = 2,
    EDGE3 = 3,
    EDGE4 = 4
};

interface ContactDetails
{
    inEdge1: Edges,
    outEdge1: Edges,
    inEdge2: Edges,
    outEdge2: Edges,
    ID: number
};

interface ContactPoint
{
    details: ContactDetails,
    pA: glm.vec2, //x, y position offset from center of body A
    pB: glm.vec2, //x, y position offset from center of body B
    n: glm.vec2,  //contact normal

    JacNormA: glm.vec3, //normal jacobian for body A
    JacNormB: glm.vec3, //normal jacobian for body B
    JacTangA: glm.vec3, //tangential jacobian for body A
    JacTangB: glm.vec3, //tangential jacobian for body B

    C0: glm.vec2,      //position constraint (normal and tangential)
    stick: boolean, //is the contact sticking (for friction)
}

//================================//
class Manifold extends Force
{
    private contacts: ContactPoint[] = []; // Maximium of two contact points
    private oldContacts: ContactPoint[] = [];
    private friction: number = 0; // Friction coefficient

    //=============== PUBLIC =================//
    constructor(bodyA: RigidBox, bodyB: RigidBox)
    {
        super(bodyA, bodyB);

        for (let i = 0; i < Force.MAX_ROWS; ++i)
        {
            this.fmax[0] = 0;
            this.fmax[2] = 0; // Max friction force is zero
        }
    }

    //================================//
    public initialize()
    {
        this.friction = Math.sqrt(this.bodyA.getFriction() * this.bodyB.getFriction());

        this.oldContacts = this.contacts;
        const oldPenalty: number[] = this.penalty;
        const oldLambda: number[] = this.lambda;
        const oldStick: boolean[] = this.contacts.map((c) => c.stick);

        // Compute new contacts
        this.contacts = []; // TODO

        // Merge contacts based on old contact info
        for(let i =0; i < this.contacts.length; ++i)
        {
            if(this.oldContacts.length > i && this.contacts[i].details.ID === this.oldContacts[i].details.ID)
            {
                this.penalty[i * 2 + 0] = oldPenalty[i * 2 + 0];
                this.penalty[i * 2 + 1] = oldPenalty[i * 2 + 1];
                this.lambda[i * 2 + 0] = oldLambda[i * 2 + 0];
                this.lambda[i * 2 + 1] = oldLambda[i * 2 + 1];
                this.contacts[i].stick = oldStick[i];

                // Static friction means we keep the same points of contact
                if(this.contacts[i].stick)
                {
                    this.contacts[i].pA = this.oldContacts[i].pA;
                    this.contacts[i].pB = this.oldContacts[i].pB;
                }
            }
        }

        // New contacts compute steps
        for(let i = 0; i < this.contacts.length; ++i)
        {
            // Friction contact based on normal and tangential
            const n: glm.vec2 = this.contacts[i].n;
            const t: glm.vec2 = glm.vec2.fromValues( n[1], -n[0] );
            const basis: glm.mat2 = glm.mat2.fromValues(    n[0], n[1], 
                                                            t[0], t[1]  );

            const rotatedAW: glm.vec2 = glm.vec2.create();
            glm.vec2.transformMat2(rotatedAW, rotationMatrix(this.bodyA.getPosition()[2]), this.contacts[i].pA);
            const rotatedBW: glm.vec2 = glm.vec2.create();
            glm.vec2.transformMat2(rotatedBW, rotationMatrix(this.bodyB.getPosition()[2]), this.contacts[i].pB);

            // Precompute constraints and derivative at C(x-). Truncated Taylor Series.
            // Jacobians are the first derivatives.
            // They are evaluated at start of step configuration x-.
            this.contacts[i].JacNormA = glm.vec3.fromValues(basis[0], basis[1], cross2(rotatedAW, n));
            this.contacts[i].JacNormB = glm.vec3.fromValues(-basis[0], -basis[1], -cross2(rotatedBW, n));
            this.contacts[i].JacTangA = glm.vec3.fromValues(basis[2], basis[3], cross2(rotatedAW, t));
            this.contacts[i].JacTangB = glm.vec3.fromValues(-basis[2], -basis[3], -cross2(rotatedBW, t));

            // Precompute constraint values at C0: computes the contact gap
            // in the basis [n, t] at the start of the step x-.
            // Collision margin slightly biases the normal gap to help robustness.

            // Equation 15 + 18, basis * (ra(x) - rb(x)) 
            const rDiff: glm.vec2 = glm.vec2.sub(
                                        glm.vec2.create(), 
                                        glm.vec2.add(glm.vec2.create(), this.bodyA.getPos2(), rotatedAW), 
                                        glm.vec2.add(glm.vec2.create(), this.bodyB.getPos2(), rotatedBW)
                                    );

            this.contacts[i].C0 = glm.vec2.transformMat2(this.contacts[i].C0, rDiff, basis); // C0 = basis * (rA - rB)
            this.contacts[i].C0 = glm.vec2.add(this.contacts[i].C0, this.contacts[i].C0, glm.vec2.fromValues(COLLISION_MARGIN, 0)); // Add small collision margin
        }

        return this.contacts.length > 0;
    }

    //================================//
    public computeConstraints(alpha: number)
    {
        for(let i = 0; i < this.contacts.length; ++i)
        {
            // Taylor series approximation in equation 18
            const diffpA: glm.vec3 = glm.vec3.sub(glm.vec3.create(), this.bodyA.getPosition(), this.bodyA.lastPosition);
            const diffpB: glm.vec3 = glm.vec3.sub(glm.vec3.create(), this.bodyB.getPosition(), this.bodyB.lastPosition);

            const alphaC0: glm.vec2 = glm.vec2.scale(glm.vec2.create(), this.contacts[i].C0, (1 - alpha));
            this.C[i * 2 + 0] = alphaC0[0] + glm.vec3.dot(this.contacts[i].JacNormA, diffpA) + glm.vec3.dot(this.contacts[i].JacNormB, diffpB); // Normal constraint
            this.C[i * 2 + 1] = alphaC0[1] + glm.vec3.dot(this.contacts[i].JacTangA, diffpA) + glm.vec3.dot(this.contacts[i].JacTangB, diffpB); // Tangential constraint

            // Update the friction bounds:
            // Coulomb friction model
            // fmin = -mu * fN
            // fmax =  mu * fN

            const bounds: number = Math.abs(this.lambda[i * 2 + 0]) * this.friction;
            this.fmax[i * 2 + 1] = bounds;
            this.fmin[i * 2 + 1] = -bounds;

            // Check if the contact should be sticking
            // Basically, are we within the coulomb friction cone
            // and is the constraint contained within a threshold,
            // if not they are sliding -> dynamic friction.
            this.contacts[i].stick = Math.abs(this.lambda[i * 2 + 1]) < bounds && Math.abs(this.contacts[i].C0[1]) < STICK_THRESHOLD;
        }
    }

    //================================//
    public computeDerivatives(body: RigidBox)
    {
        // We store the precomputed Jacobians
        for(let i = 0; i < this.contacts.length; ++i)
        {
            if(body === this.bodyA)
            {
                this.J[i * 2 + 0] = this.contacts[i].JacNormA; // Normal
                this.J[i * 2 + 1] = this.contacts[i].JacTangA; // Tangential
            }
            else
            {
                this.J[i * 2 + 0] = this.contacts[i].JacNormB; // Normal
                this.J[i * 2 + 1] = this.contacts[i].JacTangB; // Tangential
            }
        }
    }
}

export default Manifold;