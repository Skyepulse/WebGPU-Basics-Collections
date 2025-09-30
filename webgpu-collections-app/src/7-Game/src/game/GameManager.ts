/*
 * GameManager.ts
 * 
 * General manager that ties the rendering and physics process together.
 * 
 */

//================================//
import * as glm from 'gl-matrix';
import GameRenderer from "./GameRenderer";
import RigidBox from "./RigidBox";
import { rand, randomPosInRectRot, randomColorUint8 } from "@src/helpers/MathUtils";

//================================//
class GameManager
{
    private logging: boolean = true;
    private running: boolean = false;
    private rafID: number | null = null;

    private canvas: HTMLCanvasElement | null = null;

    private gameRenderer: GameRenderer;
    private rigidBoxes: RigidBox[] = [];
    // private forces: Force[] = [];

    private lastFrameTime: number = 0;

    //=============== PUBLIC =================//
    constructor(canvas: HTMLCanvasElement)
    {
        this.canvas = canvas;
        this.gameRenderer = new GameRenderer(this.canvas as HTMLCanvasElement, this);
    }

    //================================//
    public async initialize()
    {
        this.log("Hello World!");

        // Game Renderer
        await this.gameRenderer.initialize();
        this.initializeWindowEvents();

        this.startMainLoop();
    }

    //================================//
    public async cleanup()
    {
        this.log("Goodbye World!");
        this.stop();
    }

    //================================//
    public toggleLogging()
    {
        this.logging = !this.logging;
    }

    //================================//
    public stop()
    {
        if (!this.running) return;
        this.running = false;

        if (this.rafID != null)
        {
            cancelAnimationFrame(this.rafID);
            this.rafID = null;
        }
        this.log("Main loop stopped.");
    }

    //================================//
    public log(msg: string)
    {
        if (this.logging)
            console.log(`[GameManager] ${msg}`);
    }

    //================================//
    public logWarn(msg: string)
    {
        if (this.logging)
            console.warn(`[GameManager] ${msg}`);
    }

    //=============== PRIVATE =================/
    private startMainLoop()
    {
        if (this.running)
        {
            this.logWarn("Main loop already running!");
            return;
        }

        this.running = true;
        this.lastFrameTime = performance.now();

        const frame = (now: number) =>
        {
            if (!this.running) return;

            const dt = now - this.lastFrameTime;
            this.lastFrameTime = now;

            this.log("Frame diff:" + dt.toFixed(2) + "ms");

            this.gameRenderer.render();
            this.rafID = requestAnimationFrame(frame);
        }

        this.rafID = requestAnimationFrame(frame);
    }

    //================================//
    public addRigidBox(
        pos: glm.vec3 = randomPosInRectRot(0, 0, GameRenderer.xWorldSize, GameRenderer.yWorldSize), 
        scale: glm.vec2 = glm.vec2.fromValues(rand(2, 10), rand(2, 10)), 
        velocity: glm.vec3 = glm.vec3.fromValues(0, 0, 0),
        color: Uint8Array = randomColorUint8()
    ): void
    {
        const box = new RigidBox(scale, color, 1.0, 1.0, pos, velocity);

        box.id = this.gameRenderer.addInstanceBox(box);
        if (box.id !== -1) {
            this.rigidBoxes.push(box);
        } else {
            this.logWarn("Failed to add box instance to renderer.");
        }
    }

    //================================//
    public initializeWindowEvents(): void
    {
        // Add box on click
        window.addEventListener('click', (event: MouseEvent) => {
            
            // Print where in the canvas was clicked
            if (!this.canvas) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            const canvasX = (x / this.canvas.width) * GameRenderer.xWorldSize;
            const canvasY = (1.0 - (y / this.canvas.height)) * GameRenderer.yWorldSize;
            const pos = glm.vec3.fromValues(canvasX, canvasY, rand(0, Math.PI * 2));

            this.addRigidBox(pos);
        });
    }
}

//================================//
export default GameManager;