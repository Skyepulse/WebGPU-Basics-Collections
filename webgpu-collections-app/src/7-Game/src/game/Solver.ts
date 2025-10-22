import * as glm from 'gl-matrix';
import type RigidBox from "./RigidBox";
import Force from "./Force";
import Manifold, { type ContactRender } from './Manifold';
import { outerMult, solveLDLT } from '@src/helpers/MathUtils';

const PENALTY_MIN = 1;
const PENALTY_MAX = 1000000000;

//================================//
class Solver
{
    public dt: number = 0;
    public gravity: glm.vec2 = glm.vec2.fromValues(0, -9.81);
    public iterations: number = 0;

    public alpha: number = 0.99;
    public beta: number = 100000;
    public gamma: number = 0.99;

    public postStabilization: boolean = false;

    public bodies: RigidBox[] = [];
    public forces: Force[] = [];

    public contactsToRender: ContactRender[] = [];

    //============= PUBLIC ===================//
    public Clear(): void
    {
        this.bodies = [];
        this.forces = [];
    }

    //================================//
    public setDefaults(): void
    {
        this.dt = 1 / 60;
        this.gravity = glm.vec2.fromValues(0, -9.81);
        this.iterations = 10;

        this.beta = 100000;
        this.alpha = 0.99;
        this.gamma = 0.99;

        // Post stabilization applies an extra iteration to fix positional error.
        // Is used instead of Alpha.
        this.postStabilization = true;
    }

    //================================//
    public step(dt: number): void
    {
        if (Math.abs(dt - this.dt) > 0.01)
            console.warn(`Warning: Physics timestep changed from ${this.dt} to ${dt}. This may cause instability.`);
        
        this.contactsToRender = [];
        // Detection: NAIVE O(n^2) FOR NOW
        for (let i = 0; i < this.bodies.length; ++i)
        {
            for (let j = i + 1; j < this.bodies.length; ++j)
            {
                const bodyA: RigidBox = this.bodies[i];
                const bodyB: RigidBox = this.bodies[j];

                const dp: glm.vec2 = glm.vec2.sub(glm.vec2.create(), bodyA.getPos2(), bodyB.getPos2());
                const r: number = bodyA.getRadius() + bodyB.getRadius();

                if (glm.vec2.squaredLength(dp) <= r * r)
                {
                    if(!bodyA.isConstrainedTo(bodyB))
                    {
                        let newManifold: Manifold = new Manifold(bodyA, bodyB);
                        this.forces.push(newManifold);
                        bodyA.forces.push(newManifold);
                        bodyB.forces.push(newManifold);
                    }
                }
            }
        }

        // WarmStarting and Initialization
        for (let i = 0; i < this.forces.length; ++i)
        {
            const force: Force = this.forces[i];
            const isUsed: boolean = force.initialize();

            if (!isUsed)
            {
                // Remove the force if not used.
                this.forces.splice(i, 1);
                --i;

                const indexA = force.bodyA.forces.indexOf(force);
                if (indexA !== -1) force.bodyA.forces.splice(indexA, 1);

                const indexB = force.bodyB.forces.indexOf(force);
                if (indexB !== -1) force.bodyB.forces.splice(indexB, 1);

                continue;
            }

            this.contactsToRender.push(...force.getContactRenders());

            for (let j = 0; j < force.getRows(); ++j)
            {
                if (this.postStabilization)
                {
                    // REUSE THE PENALTY FROM PREVIOUS STEP by gamma
                    let newPenalty = force.penalty[j] * this.gamma;
                    if (newPenalty < PENALTY_MIN) newPenalty = PENALTY_MIN;
                    if (newPenalty > PENALTY_MAX) newPenalty = PENALTY_MAX;
                    force.penalty[j] = newPenalty;
                }
                else
                {
                    force.lambda[j] = force.lambda[j] * this.alpha * this.gamma;
                    let newPenalty = force.penalty[j] * this.gamma;
                    if (newPenalty < PENALTY_MIN) newPenalty = PENALTY_MIN;
                    if (newPenalty > PENALTY_MAX) newPenalty = PENALTY_MAX;
                    force.penalty[j] = newPenalty;
                }

                force.penalty[j] = Math.min(force.penalty[j], force.stiffness[j]);
            }
        }

         // Warmstart bodies
        for (let i = 0; i < this.bodies.length; ++i)
        {
            const body: RigidBox = this.bodies[i];

            // Constraint rotation speed
            let rotationVelocity = body.getVelocity()[2];
            if (rotationVelocity > 50) rotationVelocity = 50;
            if (rotationVelocity < -50) rotationVelocity = -50;
            body.setVelocity(glm.vec3.fromValues(body.getVelocity()[0], body.getVelocity()[1], rotationVelocity));

            // The prediction based on current velocity
            body.inertial = glm.vec3.add(glm.vec3.create(), body.getPosition(), glm.vec3.scale(glm.vec3.create(), body.getVelocity(), this.dt));
            if (body.getMass() !== 0)
            {
                let gv = glm.vec3.scale(glm.vec3.create(), glm.vec3.fromValues(this.gravity[0], this.gravity[1], 0), this.dt * this.dt);
                body.inertial = glm.vec3.add(body.inertial, body.inertial, gv);
            }

            // Adaptative warmstarting
            const acceleration: glm.vec3 = glm.vec3.scale(glm.vec3.create(), glm.vec3.sub(glm.vec3.create(), body.getVelocity(), body.getPrevVelocity()), 1 / this.dt);
            const accelExt: number = acceleration[1] * Math.sign(this.gravity[1]);
            let accelWeight: number = accelExt / Math.abs(this.gravity[1]);
            if (accelWeight < 0) accelWeight = 0;
            if (accelWeight > 1) accelWeight = 1;

            body.lastPosition = glm.vec3.clone(body.getPosition());
            const term2: glm.vec3 = glm.vec3.scale(glm.vec3.create(), body.getVelocity(), this.dt);
            const term3: glm.vec3 = glm.vec3.scale(glm.vec3.create(), glm.vec3.fromValues(this.gravity[0], this.gravity[1], 0), accelWeight * this.dt * this.dt);
            body.setPosition(glm.vec3.add(glm.vec3.create(), body.getPosition(), glm.vec3.add(glm.vec3.create(), term2, term3)));
        }

         // Main iteration loop
        const iterations = this.iterations + (this.postStabilization ? 1 : 0);

        for (let iter = 0; iter < iterations; ++iter)
        {
            let currentAlpha: number = this.alpha;
            if (this.postStabilization) currentAlpha = iter < this.iterations ? 1 : 0;

            // PRIMAL FIRST
            for (const body of this.bodies)
            {
                if (body.isStatic()) continue;

                // Equation 5 and 6 
                const M: glm.mat3 = glm.mat3.fromValues(
                    body.getMass(), 0, 0,
                    0, body.getMass(), 0,
                    0, 0, body.getMoment()
                );   
                const lhs: glm.mat3 = glm.mat3.multiplyScalar(glm.mat3.create(), M, 1 / (this.dt * this.dt));
                const rhs: glm.vec3 = glm.vec3.transformMat3(glm.vec3.create(), glm.vec3.sub(glm.vec3.create(), body.getPosition(), body.inertial), lhs);

                for (const force of body.forces)
                {
                    force.computeConstraints(currentAlpha);
                    force.computeDerivatives(body);

                    for (let j = 0; j < force.getRows(); ++j)
                    {
                        // if stifness is infinity, use lambda
                        let lambda: number = (force.stiffness[j] === Infinity) ? force.lambda[j] : 0;

                        // Clamped force magnitude
                        let f: number = force.penalty[j] * force.C[j] + lambda;
                        if (f < force.fmin[j]) f = force.fmin[j];
                        if (f > force.fmax[j]) f = force.fmax[j];

                        const G: glm.mat3 = glm.mat3.fromValues(
                            glm.vec3.length(glm.vec3.fromValues(force.H[j][0], force.H[j][3], force.H[j][6])), 0, 0,
                            0, glm.vec3.length(glm.vec3.fromValues(force.H[j][1], force.H[j][4], force.H[j][7])), 0,
                            0, 0, glm.vec3.length(glm.vec3.fromValues(force.H[j][2], force.H[j][5], force.H[j][8]))
                        );
                        glm.mat3.multiplyScalar(G, G, Math.abs(f));

                        // Accumulate forces (equation 13) and hessian (equation 17)
                        glm.vec3.add(rhs, rhs, glm.vec3.scale(glm.vec3.create(), force.J[j], f));
                        const outer: glm.mat3 = outerMult(force.J[j], glm.vec3.scale(glm.vec3.create(), force.J[j], force.penalty[j]));
                        glm.mat3.add(lhs, lhs, outer);
                        glm.mat3.add(lhs, lhs, G);
                    }
                }

                const dx: glm.vec3 = solveLDLT(lhs, rhs);
                body.setPosition(glm.vec3.sub(glm.vec3.create(), body.getPosition(), dx));
            }

            // DUAL UPDATE, except for last iteration if post-stabilization
            if (iter < this.iterations)
            {
                for (const force of this.forces)
                {
                    force.computeConstraints(currentAlpha);

                    for (let j = 0; j < force.getRows(); ++j)
                    {
                        let lambda: number = (force.stiffness[j] === Infinity) ? force.lambda[j] : 0;

                        // Update the lambda
                        force.lambda[j] = lambda + force.penalty[j] * force.C[j];
                        if (force.lambda[j] < force.fmin[j]) force.lambda[j] = force.fmin[j];
                        if (force.lambda[j] > force.fmax[j]) force.lambda[j] = force.fmax[j];

                        // If it exceeds fracture threshold, disable it.
                        if (Math.abs(force.lambda[j]) >= force.fracture[j]) force.disable();

                        // update penalty
                        if (force.lambda[j] > force.fmin[j] && force.lambda[j] < force.fmax[j])
                        {
                            force.penalty[j] = Math.min(force.penalty[j] + this.beta * Math.abs(force.C[j]), Math.min(force.stiffness[j], PENALTY_MAX));
                        }
                    }
                }
            }

            // Post stabilization
            if (iter == this.iterations - 1)
            {
                for (const body of this.bodies)
                {
                    body.prevVelocity = body.getVelocity();
                    if (body.getMass() > 0)
                    {
                        const newVelocity: glm.vec3 = glm.vec3.sub(glm.vec3.create(), body.getPosition(), body.lastPosition);
                        glm.vec3.scale(newVelocity, newVelocity, 1 / this.dt);
                        body.setVelocity(newVelocity);
                    }
                }
            }
        }
    }

    //================================//
    public addRigidBox(box: RigidBox): void
    {
        if (this.bodies.indexOf(box) === -1)
            this.bodies.push(box);
    }

    //================================//
    public removeRigidBox(box: RigidBox): void
    {
        const index = this.bodies.indexOf(box);
        if (index !== -1)
            this.bodies.splice(index, 1);
    }
}

export default Solver;