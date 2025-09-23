/*
 * GameManager.ts
 * 
 * General manager that ties the rendering and physics process together.
 * 
 */

//================================//
import GameRenderer from "./GameRenderer";

//================================//
class GameManager
{
    private logging: boolean = true;
    private running: boolean = false;
    private rafID: number | null = null;

    private canvas: HTMLCanvasElement | null = null;

    private gameRenderer: GameRenderer;

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

        const frame = (now: number) =>
        {
            if (!this.running) return;

            // this.log(`Frame @ ${now.toFixed(2)}ms`);
            this.gameRenderer.render();
            this.rafID = requestAnimationFrame(frame);
        }

        this.rafID = requestAnimationFrame(frame);
    }
}

//================================//
export default GameManager;