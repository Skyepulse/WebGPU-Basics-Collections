//================================//
import GameManager from "./src/game/GameManager";

//================================//
export async function startup_7(canvas: HTMLCanvasElement)
{
    const gameManager = new GameManager(canvas);
    await gameManager.initialize();
    return gameManager;
}