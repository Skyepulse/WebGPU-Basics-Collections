import { type Material } from "@src/helpers/GeometryUtils.ts";
import * as glm from "gl-matrix";
import { degreesToRads, radsToDegrees } from "./MathUtils";

//================================//
export function getInfoElement(): HTMLElement | null 
{
    return document.getElementById("info");
}

//================================//
export function getUtilElement(): HTMLElement | null
{
    return document.getElementById("utils");
}

//================================//
export function addUtilElementDefaults(): void
{
    const utilElement = getUtilElement();
    if (utilElement)
    {
        // NOTHING YET
    }
}

//================================//
export function cleanUtilElement(): void
{
    const utilElement = getUtilElement();

    //================================//
    if (utilElement)
    {
        while(utilElement.firstChild) 
        {
            utilElement.removeChild(utilElement.firstChild);
        }
    }

    //================================//
    // Add back the constants
    addUtilElementDefaults();
}

//================================//
export function createMaterialContextMenu(position: { x: number; y: number }, currentMaterial: Material, onApplyCallback: (newMaterial: Material) => void, onCancelCallback: () => void): HTMLDivElement
{
    const menu = document.createElement('div');
    menu.style.cssText = `
        position: fixed;
        left: ${position.x + 15}px;
        top: ${position.y}px;
        background: rgba(40, 40, 40, 0.95);
        border: 1px solid #555;
        border-radius: 8px;
        padding: 12px;
        z-index: 10000;
        min-width: 180px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        font-family: sans-serif;
        font-size: 13px;
        color: #eee;
    `;

    const title = document.createElement('div');
    title.textContent = `Material: ${currentMaterial.name}`;
    title.style.cssText = `
        font-weight: bold;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #555;
    `;
    menu.appendChild(title);

    // Albedo color picker
    const albedoSection = document.createElement('div');
    albedoSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const albedoLabel = document.createElement('label');
    albedoLabel.textContent = 'Albedo:';
    albedoSection.appendChild(albedoLabel);

    // Convert current albedo (0-1 floats) to hex color
    const toHex = (val: number) => Math.round(val * 255).toString(16).padStart(2, '0');
    const currentHex = `#${toHex(currentMaterial.albedo[0])}${toHex(currentMaterial.albedo[1])}${toHex(currentMaterial.albedo[2])}`;

    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = currentHex;
    colorPicker.style.cssText = `
        width: 50px;
        height: 30px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        padding: 0;
    `;
    colorPicker.tabIndex = -1;
    albedoSection.appendChild(colorPicker);

    // Hex value display
    const hexDisplay = document.createElement('span');
    hexDisplay.textContent = currentHex.toUpperCase();
    hexDisplay.style.cssText = 'font-family: monospace; color: #aaa;';
    albedoSection.appendChild(hexDisplay);

    colorPicker.addEventListener('input', () => {
        hexDisplay.textContent = colorPicker.value.toUpperCase();
        const r = parseInt(colorPicker.value.slice(1, 3), 16) / 255;
        const g = parseInt(colorPicker.value.slice(3, 5), 16) / 255;
        const b = parseInt(colorPicker.value.slice(5, 7), 16) / 255;
        currentMaterial.albedo = [r, g, b];
        onApplyCallback(currentMaterial);
    });

    menu.appendChild(albedoSection);

    // Metalness
    const metalnessSelection = document.createElement('div');
    metalnessSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const metalnessLabel = document.createElement('label');
    metalnessLabel.textContent = `Metalness: ${currentMaterial.metalness.toFixed(2)}`;
    metalnessSelection.appendChild(metalnessLabel);

    const metalnessSlider = document.createElement('input');
    metalnessSlider.type = 'range';
    metalnessSlider.min = '0';  
    metalnessSlider.max = '1';
    metalnessSlider.step = '0.01';
    metalnessSlider.value = currentMaterial.metalness.toString();
    metalnessSlider.style.cssText = `
        flex: 1;
        cursor: pointer;
    `;
    metalnessSlider.tabIndex = -1;
    metalnessSelection.appendChild(metalnessSlider);
    menu.appendChild(metalnessSelection);
    metalnessSlider.addEventListener('input', () => {
        const val = parseFloat(metalnessSlider.value);
        currentMaterial.metalness = isNaN(val) ? 0 : val;
        metalnessLabel.textContent = `Metalness: ${currentMaterial.metalness.toFixed(2)}`;
        onApplyCallback(currentMaterial);
    });

    const usePerlinMetalnessLabel = document.createElement('label');
    usePerlinMetalnessLabel.textContent = 'Perlin noise';
    metalnessSelection.appendChild(usePerlinMetalnessLabel);

    const usePerlinMetalnessCheckbox = document.createElement('input');
    usePerlinMetalnessCheckbox.type = 'checkbox';
    usePerlinMetalnessCheckbox.checked = currentMaterial.usePerlinMetalness;
    usePerlinMetalnessCheckbox.tabIndex = -1;
    metalnessSelection.appendChild(usePerlinMetalnessCheckbox);
    usePerlinMetalnessCheckbox.addEventListener('change', () => {
        currentMaterial.usePerlinMetalness = usePerlinMetalnessCheckbox.checked;
        onApplyCallback(currentMaterial);
    });

    // Roughness
    const roughnessSelection = document.createElement('div');
    roughnessSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const roughnessLabel = document.createElement('label');
    roughnessLabel.textContent = `Roughness: ${currentMaterial.roughness.toFixed(2)}`;
    roughnessSelection.appendChild(roughnessLabel);

    const roughnessSlider = document.createElement('input');
    roughnessSlider.type = 'range';
    roughnessSlider.min = '0';
    roughnessSlider.max = '1';
    roughnessSlider.step = '0.01';
    roughnessSlider.value = currentMaterial.roughness.toString();
    roughnessSlider.style.cssText = `
        flex: 1;
        cursor: pointer;
    `;
    roughnessSlider.tabIndex = -1;
    roughnessSelection.appendChild(roughnessSlider);
    menu.appendChild(roughnessSelection);
    roughnessSlider.addEventListener('input', () => {
        const val = parseFloat(roughnessSlider.value);
        currentMaterial.roughness = isNaN(val) ? 0 : val;
        roughnessLabel.textContent = `Roughness: ${currentMaterial.roughness.toFixed(2)}`;
        onApplyCallback(currentMaterial);
    });

    const usePerlinRoughnessLabel = document.createElement('label');
    usePerlinRoughnessLabel.textContent = 'Perlin noise';
    roughnessSelection.appendChild(usePerlinRoughnessLabel);

    const usePerlinRoughnessCheckbox = document.createElement('input');
    usePerlinRoughnessCheckbox.type = 'checkbox';
    usePerlinRoughnessCheckbox.checked = currentMaterial.usePerlinRoughness;
    usePerlinRoughnessCheckbox.tabIndex = -1;
    roughnessSelection.appendChild(usePerlinRoughnessCheckbox);
    usePerlinRoughnessCheckbox.addEventListener('change', () => {
        currentMaterial.usePerlinRoughness = usePerlinRoughnessCheckbox.checked;
        onApplyCallback(currentMaterial);
    });

    // perlin frequency
    const perlinFreqSelection = document.createElement('div');
    perlinFreqSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const perlinFreqLabel = document.createElement('label');
    perlinFreqLabel.textContent = `Perlin Frequency: ${currentMaterial.perlinFreq.toFixed(2)}`;
    perlinFreqSelection.appendChild(perlinFreqLabel);

    const perlinFreqSlider = document.createElement('input');
    perlinFreqSlider.type = 'range';
    perlinFreqSlider.min = '0.1';
    perlinFreqSlider.max = '10';
    perlinFreqSlider.step = '0.1';
    perlinFreqSlider.value = currentMaterial.perlinFreq.toString();
    perlinFreqSlider.style.cssText = `
        flex: 1;
        cursor: pointer;
    `;
    perlinFreqSlider.tabIndex = -1;
    perlinFreqSelection.appendChild(perlinFreqSlider);
    menu.appendChild(perlinFreqSelection);
    perlinFreqSlider.addEventListener('input', () => {
        const val = parseFloat(perlinFreqSlider.value);
        currentMaterial.perlinFreq = isNaN(val) ? 0.1 : val;
        perlinFreqLabel.textContent = `Perlin Frequency: ${currentMaterial.perlinFreq.toFixed(2)}`;
        onApplyCallback(currentMaterial);
    });

    // Buttons row
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `
        padding: 6px 16px;
        background: #555;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        font-size: 13px;
    `;
    cancelButton.tabIndex = -1;
    cancelButton.addEventListener('click', () => {
        onCancelCallback();
    });

    buttonsRow.appendChild(cancelButton);
    menu.appendChild(buttonsRow);

    return menu;
}

export interface SpotLight
{
    position: glm.vec3;
    intensity: number;
    direction: glm.vec3;
    coneAngle: number;
    color: glm.vec3;
    enabled: boolean;
};

//================================//
export function createLightContextMenu(position: { x: number; y: number }, currentLight: SpotLight, title: string, onApplyCallback: (light: SpotLight) => void, onCancelCallback: () => void): HTMLDivElement
{
    const menu = document.createElement('div');
    menu.style.cssText = `
        position: fixed;
        left: ${position.x + 15}px;
        top: ${position.y}px;
        background: rgba(40, 40, 40, 0.95);
        border: 1px solid #555;
        border-radius: 8px;
        padding: 12px;
        z-index: 10000;
        min-width: 180px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        font-family: sans-serif;
        font-size: 13px;
        color: #eee;
    `;

    const titlediv = document.createElement('div');
    titlediv.textContent = title;
    titlediv.style.cssText = `
        font-weight: bold;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #555;
    `;
    menu.appendChild(titlediv);

    // Enabled checkbox
    const enabledSelection = document.createElement('div');
    enabledSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const enabledLabel = document.createElement('label');
    enabledLabel.textContent = 'Enabled:';
    enabledSelection.appendChild(enabledLabel);

    const enabledCheckbox = document.createElement('input');
    enabledCheckbox.type = 'checkbox';
    enabledCheckbox.checked = currentLight.enabled;
    enabledCheckbox.tabIndex = -1;
    enabledSelection.appendChild(enabledCheckbox);
    enabledCheckbox.addEventListener('change', () => {
        currentLight.enabled = enabledCheckbox.checked;
        onApplyCallback(currentLight);
    });
    menu.appendChild(enabledSelection);

    // Position (x, y, z) picker
    const positionSelection = document.createElement('div');
    positionSelection.style.cssText = 'display: flex; flex-direction: row; gap: 6px; margin-bottom: 12px;';

    const positionLabel = document.createElement('label');
    positionLabel.textContent = 'Light position:';
    positionSelection.appendChild(positionLabel);

    ['X', 'Y', 'Z'].forEach((axis, index) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.value = currentLight.position[index].toFixed(2);
        input.style.cssText = `
            width: 50px;
            padding: 4px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #222;
            color: #eee;
        `;
        input.tabIndex = -1;
        positionSelection.appendChild(input);
        input.addEventListener('input', () => {
            const val = parseFloat(input.value);
            currentLight.position[index] = isNaN(val) ? 0 : val;
            onApplyCallback(currentLight);
        });
        input.placeholder = axis;
    });
    menu.appendChild(positionSelection);

    // Direction (x, y, z) picker
    const directionSelection = document.createElement('div');
    directionSelection.style.cssText = 'display: flex; flex-direction: row; gap: 6px; margin-bottom: 12px;';
    const directionLabel = document.createElement('label');
    directionLabel.textContent = 'Light direction:';
    directionSelection.appendChild(directionLabel);

    ['X', 'Y', 'Z'].forEach((axis, index) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.value = currentLight.direction[index].toFixed(2);
        input.style.cssText = `
            width: 50px;
            padding: 4px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #222;
            color: #eee;
        `;
        input.tabIndex = -1;
        directionSelection.appendChild(input);
        input.addEventListener('input', () => {
            const val = parseFloat(input.value);
            currentLight.direction[index] = isNaN(val) ? 0 : val;
            onApplyCallback(currentLight);
        });
        input.placeholder = axis;
    });
    menu.appendChild(directionSelection);

    // Intensity picker
    const intensitySelection = document.createElement('div');
    intensitySelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const intensityLabel = document.createElement('label');
    intensityLabel.textContent = 'Light intensity:';
    intensitySelection.appendChild(intensityLabel);

    const intensityInput = document.createElement('input');
    intensityInput.type = 'number';
    intensityInput.value = currentLight.intensity.toFixed(2);
    intensityInput.style.cssText = `
        width: 80px;
        padding: 4px;
        border: 1px solid #555;
        border-radius: 4px;
        background: #222;
        color: #eee;
    `;
    intensityInput.tabIndex = -1;
    intensitySelection.appendChild(intensityInput);
    intensityInput.addEventListener('input', () => {
        const val = parseFloat(intensityInput.value);
        currentLight.intensity = isNaN(val) ? 0 : val;
        onApplyCallback(currentLight);
    });
    menu.appendChild(intensitySelection);

    // Cone angle picker
    const coneAngleSelection = document.createElement('div');
    coneAngleSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const coneAngleLabel = document.createElement('label');
    coneAngleLabel.textContent = 'Cone angle:';
    coneAngleSelection.appendChild(coneAngleLabel);

    const coneAngleInput = document.createElement('input');
    coneAngleInput.type = 'range';
    coneAngleInput.value = radsToDegrees(currentLight.coneAngle).toFixed(2);
    coneAngleInput.min = '0';
    coneAngleInput.max = '180';
    coneAngleInput.style.cssText = `
        width: 80px;
        padding: 4px;
        border: 1px solid #555;
        border-radius: 4px;
        background: #222;
        color: #eee;
    `;
    coneAngleInput.tabIndex = -1;
    coneAngleSelection.appendChild(coneAngleInput);
    coneAngleInput.addEventListener('input', () => {
        const val = parseFloat(coneAngleInput.value);
        const rads = degreesToRads(val);
        currentLight.coneAngle = isNaN(rads) ? 0 : rads;
        onApplyCallback(currentLight);
    });
    menu.appendChild(coneAngleSelection);

    // Color picker
    const lightSelection = document.createElement('div');
    lightSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const colorLabel = document.createElement('label');
    colorLabel.textContent = 'Light color:';
    lightSelection.appendChild(colorLabel);

    // Convert current albedo (0-1 floats) to hex color
    const toHex = (val: number) => Math.round(val * 255).toString(16).padStart(2, '0');
    const currentHex = `#${toHex(currentLight.color[0])}${toHex(currentLight.color[1])}${toHex(currentLight.color[2])}`;

    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = currentHex;
    colorPicker.style.cssText = `
        width: 50px;
        height: 30px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        padding: 0;
    `;
    colorPicker.tabIndex = -1;
    lightSelection.appendChild(colorPicker);

    // Hex value display
    const hexDisplay = document.createElement('span');
    hexDisplay.textContent = currentHex.toUpperCase();
    hexDisplay.style.cssText = 'font-family: monospace; color: #aaa;';
    lightSelection.appendChild(hexDisplay);
    colorPicker.addEventListener('input', () => {
        hexDisplay.textContent = colorPicker.value.toUpperCase();
        currentLight.color = [
            parseInt(colorPicker.value.slice(1, 3), 16) / 255,
            parseInt(colorPicker.value.slice(3, 5), 16) / 255,
            parseInt(colorPicker.value.slice(5, 7), 16) / 255
        ];
        onApplyCallback(currentLight);
    });
    menu.appendChild(lightSelection);

    // Buttons row
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `
        padding: 6px 16px;
        background: #555;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        font-size: 13px;
    `;
    cancelButton.tabIndex = -1;
    cancelButton.addEventListener('click', () => {
        onCancelCallback();
    });

    buttonsRow.appendChild(cancelButton);
    menu.appendChild(buttonsRow);

    return menu;
}