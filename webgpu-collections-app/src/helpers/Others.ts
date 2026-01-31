import { type Material } from "@src/helpers/GeometryUtils.ts";
import * as glm from "gl-matrix";

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
    });

    menu.appendChild(albedoSection);

    // Buttons row
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

    const applyButton = document.createElement('button');
    applyButton.textContent = 'Apply';
    applyButton.style.cssText = `
        padding: 6px 16px;
        background: #4a9eff;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        font-size: 13px;
    `;
    applyButton.tabIndex = -1;
    applyButton.addEventListener('click', () => {
        const hex = colorPicker.value;
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;

        const newMaterial: Material = {
            name: currentMaterial.name,
            albedo: [r, g, b]
        };
        onApplyCallback(newMaterial);
    });

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
    buttonsRow.appendChild(applyButton);
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
    });
    menu.appendChild(intensitySelection);

    // Cone angle picker
    const coneAngleSelection = document.createElement('div');
    coneAngleSelection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const coneAngleLabel = document.createElement('label');
    coneAngleLabel.textContent = 'Cone angle:';
    coneAngleSelection.appendChild(coneAngleLabel);

    const coneAngleInput = document.createElement('input');
    coneAngleInput.type = 'number';
    coneAngleInput.value = currentLight.coneAngle.toFixed(2);
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
        currentLight.coneAngle = isNaN(val) ? 0 : val;
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
    });
    menu.appendChild(lightSelection);

    // Buttons row
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';

    const applyButton = document.createElement('button');
    applyButton.textContent = 'Apply';
    applyButton.style.cssText = `
        padding: 6px 16px;
        background: #4a9eff;
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        font-size: 13px;
    `;
    applyButton.tabIndex = -1;
    applyButton.addEventListener('click', () => {
        const hex = colorPicker.value;
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;

        const newLight: SpotLight = {
            position: currentLight.position,
            intensity: currentLight.intensity,
            direction: currentLight.direction,
            coneAngle: currentLight.coneAngle,
            color: [r, g, b],
            enabled: currentLight.enabled
        };
        onApplyCallback(newLight);
    });

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
    buttonsRow.appendChild(applyButton);
    menu.appendChild(buttonsRow);

    return menu;
}