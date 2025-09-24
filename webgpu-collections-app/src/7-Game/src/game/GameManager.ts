/*
 * GameManager.ts
 * 
 * General manager that ties the rendering and physics process together.
 * 
 */

//================================//
import GameRenderer from "./GameRenderer";
import { rand, randomPosInRect, randomColorUint8 } from "@src/helpers/MathUtils";

interface GameObject {
    position: Float32Array; // vec2f
    scale: Float32Array; // vec2f
    color: Uint8Array; // vec4u8
    id: number;
}

//================================//
class GameManager
{
    private logging: boolean = true;
    private running: boolean = false;
    private rafID: number | null = null;

    private canvas: HTMLCanvasElement | null = null;

    private gameRenderer: GameRenderer;
    private gameObjects: GameObject[] = [];

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

        for (let i = 0; i < 1000; i++)
        {
            const pos = randomPosInRect(0, 0, 100, 50);
            const color = randomColorUint8();
            const scale = new Float32Array([rand(0.5, 2), rand(0.5, 2)]);

            const id = this.gameRenderer.addInstance(pos, scale, color);
            if (id !== null) {
                this.gameObjects.push({ position: pos, scale: scale, color: color, id: id });
            }
        }

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

            for (let obj of this.gameObjects)
            {
                obj.position[0] += dt * 0.1;
                if (obj.position[0] > 100) {
                    obj.position[0] = 0;
                }
                this.gameRenderer.updateInstancePosition(obj.id, obj.position);
            }

            // this.log(`Frame @ ${now.toFixed(2)}ms`);
            this.gameRenderer.render();
            this.rafID = requestAnimationFrame(frame);
        }

        this.rafID = requestAnimationFrame(frame);
    }
}

//================================//
export default GameManager;