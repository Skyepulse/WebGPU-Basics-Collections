/*
 * Manifold.ts
 *
 * Represents a manifold constraint between two bodies. This mean two points of contact,
 * each with its own normal and friction coefficient. Each contact has two constraint rows.
 * One for the normal force and one for the friction (tangential) force.
 *
 * Based on Chris Gile's avbd-demo2d C++ implementation, collisions copied and modified from box2D-lite: https://github.com/erincatto/box2d-lite.
 */

// Box vertex and edge numbering:
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

//============== STRUCTS ==================//
enum Edges
{
    NO_EDGE = 0,
    EDGE1   = 1,
    EDGE2   = 2,
    EDGE3   = 3,
    EDGE4   = 4
};

//================================//
enum Axis
{
	FACE_A_X = 1,
	FACE_A_Y = 2,
	FACE_B_X = 3,
	FACE_B_Y = 4
};

//================================//
interface ContactDetails
{
    inEdge1: Edges,
    outEdge1: Edges,
    inEdge2: Edges,
    outEdge2: Edges,
    ID: number
};

//================================//
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
export interface ContactRender {
    pos: glm.vec2;
}

//================================//
interface ClipVertex
{
    cd: ContactDetails,
    v: glm.vec2
}

//============== STATIC METHODS ==================//
const createDefaultContactDetails = (): ContactDetails => ({
    inEdge1: Edges.NO_EDGE,
    outEdge1: Edges.NO_EDGE,
    inEdge2: Edges.NO_EDGE,
    outEdge2: Edges.NO_EDGE,
    ID: 0
});

//================================//
/* Flips the contact details so that body1 and body2 are swapped */
const Flip = (cd: ContactDetails): void =>
{
    const tempIn: Edges = cd.inEdge1;
    cd.inEdge1 = cd.inEdge2;
    cd.inEdge2 = tempIn;

    const tempOut: Edges = cd.outEdge1;
    cd.outEdge1 = cd.outEdge2;
    cd.outEdge2 = tempOut;
}

//================================//
function cloneDetails(cd: ContactDetails): ContactDetails {
  return {
    inEdge1: cd.inEdge1,
    outEdge1: cd.outEdge1,
    inEdge2: cd.inEdge2,
    outEdge2: cd.outEdge2,
    ID: cd.ID
  };
}

//================================//
function packFeatureID(cd: ContactDetails): number {
  // pack four 8-bit edge tags into a 32-bit int
  return ((cd.inEdge1  & 0xFF)      ) |
         ((cd.outEdge1 & 0xFF) << 8 ) |
         ((cd.inEdge2  & 0xFF) << 16) |
         ((cd.outEdge2 & 0xFF) << 24);
}

//================================//
function makeEmptyContact(): ContactPoint {
  return {
    details: createDefaultContactDetails(),
    pA: glm.vec2.create(),
    pB: glm.vec2.create(),
    n:  glm.vec2.create(),
    JacNormA: glm.vec3.create(),
    JacNormB: glm.vec3.create(),
    JacTangA: glm.vec3.create(),
    JacTangB: glm.vec3.create(),
    C0: glm.vec2.create(),
    stick: false
  };
}

//================================//
/* Clips a segment to line, meaning return the portion of the segment on the side of the line defined by normal and offset
 * Sutherland–Hodgman clipping algorithm
 * Selects vertices on the visible side of the line
 * https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
 */
const ClipSegmentToLine = (vOut: ClipVertex[], vIn: ClipVertex[], normal: glm.vec2, offset: number, clipEdge: Edges): number =>
{
    let numOut: number = 0;

    // Distance of end points to the line
    const d0: number = glm.vec2.dot(normal, vIn[0].v) - offset;
    const d1: number = glm.vec2.dot(normal, vIn[1].v) - offset;

    // Are we behind or in front of the plane?
    if (d0 <= 0) vOut[numOut++] = { v: glm.vec2.clone(vIn[0].v), cd: cloneDetails(vIn[0].cd) }; // vIn[0] is inside
    if (d1 <= 0) vOut[numOut++] = { v: glm.vec2.clone(vIn[1].v), cd: cloneDetails(vIn[1].cd) }; // vIn[1] is inside

    // If we have opposite side of the plane (i.e the lines intersect)
    if (d0 * d1 < 0) // Less than zero means they are on opposite sides of the line
    {
        // numOut has to be 1 here
        const interp: number = d0 / (d0 - d1);
        const v = glm.vec2.lerp(glm.vec2.create(), vIn[0].v, vIn[1].v, interp);

        let cd = cloneDetails(d0 > 0 ? vIn[0].cd : vIn[1].cd);
        if (d0 > 0) // vIn[0] is outside
        {
            cd.inEdge1  = clipEdge;
            cd.inEdge2  = Edges.NO_EDGE;
        }
        else    
        { // vIn[1] is outside
            cd.outEdge1 = clipEdge;
            cd.outEdge2 = Edges.NO_EDGE;
        }
        cd.ID = packFeatureID(cd);

        vOut[numOut++] = { v, cd };
    }

    return numOut;
}

//================================//
/* Incident edge computation
 * In SAT-based box–box contact, once you’ve chosen the reference face
 * (and its outward normal), you need from the other box the edge whose outward normal is most 
 * opposite to that reference normal. Those two edge endpoints become the segment you’ll later clip 
 * to the reference face’s side planes.
 */
const ComputeIncidentEdge = (c: ClipVertex[], h: glm.vec2, pos: glm.vec2, rot: glm.mat2, normal: glm.vec2): void =>
{
    const rotT = glm.mat2.transpose(glm.mat2.create(), rot);
    const n: glm.vec2 = glm.vec2.transformMat2(glm.vec2.create(), normal, rotT); // Normal in local space of box B
    glm.vec2.scale(n, n, -1);
    const absN: glm.vec2 = glm.vec2.fromValues(Math.abs(n[0]), Math.abs(n[1])); // Absolute normal

    const biggerX: boolean = absN[0] > absN[1]; // Which axis is most aligned with the normal

    if (biggerX)
    {
        if (n[0] > 0) // Meaning normal is pointing in positive x direction
        {
            c[0].v = glm.vec2.fromValues(h[0], -h[1]);
            c[0].cd.inEdge2 = Edges.EDGE3;
            c[0].cd.outEdge2 = Edges.EDGE4;

            c[1].v = glm.vec2.fromValues(h[0], h[1]);
            c[1].cd.inEdge2 = Edges.EDGE4;
            c[1].cd.outEdge2 = Edges.EDGE1;
        }
        else
        {
            c[0].v = glm.vec2.fromValues(-h[0], h[1]);
            c[0].cd.inEdge2 = Edges.EDGE1;
            c[0].cd.outEdge2 = Edges.EDGE2;

            c[1].v = glm.vec2.fromValues(-h[0], -h[1]);
            c[1].cd.inEdge2 = Edges.EDGE2;
            c[1].cd.outEdge2 = Edges.EDGE3;
        }
    }
    else
    {
        if (n[1] > 0) // Meaning normal is pointing in positive y direction
        {
            c[0].v = glm.vec2.fromValues(h[0], h[1]);
            c[0].cd.inEdge2 = Edges.EDGE4;
            c[0].cd.outEdge2 = Edges.EDGE1;

            c[1].v = glm.vec2.fromValues(-h[0], h[1]);
            c[1].cd.inEdge2 = Edges.EDGE1;
            c[1].cd.outEdge2 = Edges.EDGE2;
        }
        else
        {
            c[0].v = glm.vec2.fromValues(-h[0], -h[1]);
            c[0].cd.inEdge2 = Edges.EDGE2;
            c[0].cd.outEdge2 = Edges.EDGE3;

            c[1].v = glm.vec2.fromValues(h[0], -h[1]);
            c[1].cd.inEdge2 = Edges.EDGE3;
            c[1].cd.outEdge2 = Edges.EDGE4;
        }
    }

    c[0].v = glm.vec2.add(glm.vec2.create(), pos, glm.vec2.transformMat2(glm.vec2.create(), c[0].v, rot));
    c[1].v = glm.vec2.add(glm.vec2.create(), pos, glm.vec2.transformMat2(glm.vec2.create(), c[1].v, rot));
}

//================================//
class Manifold extends Force
{
    private contacts: ContactPoint[] = []; // Maximium of two contact points
    private numContacts: number = 0;
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

            this.fmin[0] = -Infinity;
            this.fmin[2] = -Infinity;
        }
    }

    //================================//
    public initialize()
    {
        this.friction = Math.sqrt(this.bodyA.getFriction() * this.bodyB.getFriction());

        this.oldContacts = this.contacts.slice();
        const oldPenalty: number[] = this.penalty.slice();
        const oldLambda: number[] = this.lambda.slice();
        const oldStick: boolean[] = this.oldContacts.map((c) => c.stick);

        // Compute new contacts
        this.contacts.length = 0;
        const numContacts: number = Manifold.collide(this.bodyA, this.bodyB, this.contacts);
        this.numContacts = numContacts;

        // Merge contacts based on old contact info
        for (let i = 0; i < this.contacts.length; ++i) {
            // default warmstart state
            this.penalty[i*2+0] = 0; this.penalty[i*2+1] = 0;
            this.lambda[i*2+0]  = 0; this.lambda[i*2+1]  = 0;
            this.contacts[i].stick = false;

            const id = this.contacts[i].details.ID;
            const j = this.oldContacts.findIndex(oc => oc.details.ID === id);
            if (j !== -1) {
                this.penalty[i*2+0] = oldPenalty[j*2+0];
                this.penalty[i*2+1] = oldPenalty[j*2+1];
                this.lambda[i*2+0]  = oldLambda[j*2+0];
                this.lambda[i*2+1]  = oldLambda[j*2+1];
                this.contacts[i].stick = oldStick[j];

                // If sticking, reuse the old local offsets for better static friction
                if (this.contacts[i].stick) {
                this.contacts[i].pA = glm.vec2.clone(this.oldContacts[j].pA);
                this.contacts[i].pB = glm.vec2.clone(this.oldContacts[j].pB);
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

            const rotatedAW: glm.vec2 = glm.vec2.transformMat2(glm.vec2.create(), this.contacts[i].pA, rotationMatrix(this.bodyA.getPosition()[2]));
            const rotatedBW: glm.vec2 = glm.vec2.transformMat2(glm.vec2.create(), this.contacts[i].pB, rotationMatrix(this.bodyB.getPosition()[2]));

            // Precompute constraints and derivative at C(x-). Truncated Taylor Series.
            // Jacobians are the first derivatives.
            // They are evaluated at start of step configuration x-.
            this.contacts[i].JacNormA = glm.vec3.fromValues(basis[0], basis[2], cross2(rotatedAW, n));
            this.contacts[i].JacNormB = glm.vec3.fromValues(-basis[0], -basis[2], -cross2(rotatedBW, n));
            this.contacts[i].JacTangA = glm.vec3.fromValues(basis[1], basis[3], cross2(rotatedAW, t));
            this.contacts[i].JacTangB = glm.vec3.fromValues(-basis[1], -basis[3], -cross2(rotatedBW, t));

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

    //================================//
    public static collide(boxA: RigidBox, boxB: RigidBox, contacts: ContactPoint[]): number
    {
        contacts.length = 0; // Mutate the array
        let n = glm.vec2.create();

        const RA = rotationMatrix(boxA.getPosition()[2]); // R(A): world-from-local
        const RB = rotationMatrix(boxB.getPosition()[2]); // R(B): world-from-local
        const RtA = glm.mat2.transpose(glm.mat2.create(), RA); // world→local
        const RtB = glm.mat2.transpose(glm.mat2.create(), RB); // world→local

        // Half Extents
        const hA = glm.vec2.scale(glm.vec2.create(), boxA.getScale(), 0.5);
        const hB = glm.vec2.scale(glm.vec2.create(), boxB.getScale(), 0.5);

        const posA = boxA.getPos2();
        const posB = boxB.getPos2();

        const rotA = boxA.getRotationMatrix();
        const rotB = boxB.getRotationMatrix();

        const diff = glm.vec2.sub(glm.vec2.create(), posB, posA);

        // center to center vector in each's local frame
        const dA = glm.vec2.transformMat2(glm.vec2.create(), diff, RtA);
        const dB = glm.vec2.transformMat2(glm.vec2.create(), diff, RtB);
        
        // Cheap SAT bounds
        const absDA = glm.vec2.fromValues(Math.abs(dA[0]), Math.abs(dA[1]));
        const absDB = glm.vec2.fromValues(Math.abs(dB[0]), Math.abs(dB[1]));

        const C: glm.mat2 = glm.mat2.multiply(glm.mat2.create(), RtA, rotB); // This C is relative rotation matrix from A to B
        const absC: glm.mat2 = glm.mat2.fromValues( Math.abs(C[0]), Math.abs(C[1]),
                                                    Math.abs(C[2]), Math.abs(C[3]) );
        const absCT: glm.mat2 = glm.mat2.transpose(glm.mat2.create(), absC);

        // Faces penetration computation
        const faceA = glm.vec2.sub(glm.vec2.create(), absDA, glm.vec2.add(glm.vec2.create(), hA, glm.vec2.transformMat2(glm.vec2.create(), hB, absC)));
        const faceB = glm.vec2.sub(glm.vec2.create(), absDB, glm.vec2.add(glm.vec2.create(), hB, glm.vec2.transformMat2(glm.vec2.create(), hA, absCT)));

        if (faceA[0] > 0 || faceA[1] > 0 || faceB[0] > 0 || faceB[1] > 0)
            return 0; // No collision

        let bestAxis: Axis;
        let separation: number;

        // BOX A default case
        bestAxis = Axis.FACE_A_X;
        separation = faceA[0];
        if (dA[0] > 0) n = glm.vec2.fromValues(rotA[0], rotA[1]); // Positive x axis, rotA column 0
        else n = glm.vec2.fromValues(-rotA[0], -rotA[1]); // Negative x axis

        const relativeTol: number = 0.95;
        const absoluteTol: number = 0.01;

        if (faceA[1] > relativeTol * separation + absoluteTol * hA[1])
        {
            bestAxis = Axis.FACE_A_Y;
            separation = faceA[1];
            if (dA[1] > 0) n = glm.vec2.fromValues(rotA[2], rotA[3]); // Positive y axis, rotA column 1
            else n = glm.vec2.fromValues(-rotA[2], -rotA[3]);
        }

        if (faceB[0] > relativeTol * separation + absoluteTol * hB[0])
        {
            bestAxis = Axis.FACE_B_X;
            separation = faceB[0];
            if (dB[0] > 0) n = glm.vec2.fromValues(rotB[0], rotB[1]); // Positive x axis, rotB column 0
            else n = glm.vec2.fromValues(-rotB[0], -rotB[1]);
        }

        if (faceB[1] > relativeTol * separation + absoluteTol * hB[1])
        {
            bestAxis = Axis.FACE_B_Y;
            separation = faceB[1];
            if (dB[1] > 0) n = glm.vec2.fromValues(rotB[2], rotB[3]); // Positive y axis, rotB column 1
            else n = glm.vec2.fromValues(-rotB[2], -rotB[3]);
        }

        // Now that we have the separating axis,
        // set up the clipping plane

        let frontNormal: glm.vec2;
        let sideNormal: glm.vec2;
        const incidentEdge: ClipVertex[] = [{cd: createDefaultContactDetails(), v: glm.vec2.create()}, {cd: createDefaultContactDetails(), v: glm.vec2.create()}];
        let front: number;
        let negSide: number;
        let posSide: number;
        let negEdge: Edges = Edges.NO_EDGE;
        let posEdge: Edges = Edges.NO_EDGE;

        let side: number;
        switch(bestAxis)
        {
            case Axis.FACE_A_X: // First default case
                frontNormal = n;
                front = glm.vec2.dot(posA, frontNormal) + hA[0];
                sideNormal = glm.vec2.fromValues(rotA[2], rotA[3]); // RotA column 1
                side = glm.vec2.dot(posA, sideNormal);
                negSide = -side + hA[1];
                posSide =  side + hA[1];
                negEdge = Edges.EDGE3; // Direction of negative normal edge
                posEdge = Edges.EDGE1;
                ComputeIncidentEdge(incidentEdge, hB, posB, rotB, frontNormal);
              break;
            case Axis.FACE_A_Y:
                frontNormal = n;
                front = glm.vec2.dot(posA, frontNormal) + hA[1];
                sideNormal = glm.vec2.fromValues(rotA[0], rotA[1]);
                side = glm.vec2.dot(posA, sideNormal);
                negSide = -side + hA[0];
                posSide =  side + hA[0];
                negEdge = Edges.EDGE2;
                posEdge = Edges.EDGE4;
                ComputeIncidentEdge(incidentEdge, hB, posB, rotB, frontNormal);
              break;
            case Axis.FACE_B_X:
                frontNormal = glm.vec2.scale(glm.vec2.create(), n, -1);
                front = glm.vec2.dot(posB, frontNormal) + hB[0];
                sideNormal = glm.vec2.fromValues(rotB[2], rotB[3]);
                side = glm.vec2.dot(posB, sideNormal);
                negSide = -side + hB[1];
                posSide =  side + hB[1];
                negEdge = Edges.EDGE3;
                posEdge = Edges.EDGE1;
                ComputeIncidentEdge(incidentEdge, hA, posA, rotA, frontNormal);
              break;
            case Axis.FACE_B_Y:
                frontNormal = glm.vec2.scale(glm.vec2.create(), n, -1);
                front = glm.vec2.dot(posB, frontNormal) + hB[1];
                sideNormal = glm.vec2.fromValues(rotB[0], rotB[1]);
                side = glm.vec2.dot(posB, sideNormal);
                negSide = -side + hB[0];
                posSide =  side + hB[0];
                negEdge = Edges.EDGE2;
                posEdge = Edges.EDGE4;
                ComputeIncidentEdge(incidentEdge, hA, posA, rotA, frontNormal);
              break;
        }    
        
        // We then clip other face with 1 face plane, and 4 edge planes

        const cp1: ClipVertex[] = [{cd: createDefaultContactDetails(), v: glm.vec2.create()}, {cd: createDefaultContactDetails(), v: glm.vec2.create()}];
        const cp2: ClipVertex[] = [{cd: createDefaultContactDetails(), v: glm.vec2.create()}, {cd: createDefaultContactDetails(), v: glm.vec2.create()}];
        let np: number;

        // Clip to Side 1 of box
        np = ClipSegmentToLine(cp1, incidentEdge, glm.vec2.scale(glm.vec2.create(), sideNormal, -1), negSide, negEdge);
        
        if (np < 2) // Meaning the two points were in front of plane, no collision
            return 0;

        // Clip to neg box side 1
        np = ClipSegmentToLine(cp2, cp1, sideNormal, posSide, posEdge);

        if (np < 2) // Meaning the two points were in front of plane, no collision
            return 0;

        contacts.push(makeEmptyContact(), makeEmptyContact());

        let numContacts: number = 0;
        for(let i = 0; i < 2; ++i)
        {
            const sep: number = glm.vec2.dot(frontNormal, cp2[i].v) - front;

            if (sep <= 0) 
            {
                const dst = contacts[numContacts];
                dst.n = glm.vec2.scale(glm.vec2.create(), n, -1);

                const isBFace = (bestAxis === Axis.FACE_B_X || bestAxis === Axis.FACE_B_Y);

                // Compute: clipPoint - frontNormal * sep
                const slidPoint = glm.vec2.sub(
                    glm.vec2.create(), 
                    cp2[i].v, 
                    glm.vec2.scale(glm.vec2.create(), frontNormal, sep)
                );

                if (!isBFace) 
                {
                    // A-face reference
                    dst.pA = glm.vec2.transformMat2(glm.vec2.create(), glm.vec2.sub(glm.vec2.create(), slidPoint, posA), RtA);
                    dst.pB = glm.vec2.transformMat2(glm.vec2.create(), glm.vec2.sub(glm.vec2.create(), cp2[i].v, posB), RtB);
                    dst.details = cloneDetails(cp2[i].cd);
                } 
                else 
                {
                    // B-face reference
                    dst.pA = glm.vec2.transformMat2(glm.vec2.create(), glm.vec2.sub(glm.vec2.create(), cp2[i].v, posA), RtA);
                    dst.pB = glm.vec2.transformMat2(glm.vec2.create(), glm.vec2.sub(glm.vec2.create(), slidPoint, posB), RtB);
                    
                    let det = cloneDetails(cp2[i].cd);
                    Flip(det);
                    dst.details = det;
                }

                dst.details.ID = packFeatureID(dst.details);
                ++numContacts;
                if (numContacts === 2) break;
            }
        }

        contacts.length = numContacts;
        return numContacts;
    }

    //================================//
    public getContactRenders(): ContactRender[] 
    {
        const renders: ContactRender[] = [];

        const RA = rotationMatrix(this.bodyA.getPosition()[2]);
        const RB = rotationMatrix(this.bodyB.getPosition()[2]);
        const PA = this.bodyA.getPos2();
        const PB = this.bodyB.getPos2();

        for (let i = 0; i < this.numContacts; ++i)
        {
            // world position on A: xA + R(A) * pA
            const worldPosA = glm.vec2.add(
                glm.vec2.create(),
                PA,
                glm.vec2.transformMat2(glm.vec2.create(), this.contacts[i].pA, RA)
            );
            renders.push({ pos: worldPosA });

            // world position on B: xB + R(B) * pB
            const worldPosB = glm.vec2.add(
                glm.vec2.create(),
                PB,
                glm.vec2.transformMat2(glm.vec2.create(), this.contacts[i].pB, RB)
            );
            renders.push({ pos: worldPosB });
        }

        return renders;
    }

    //================================//
    public getRows(): number { return this.contacts.length * 2; }
}

export default Manifold;