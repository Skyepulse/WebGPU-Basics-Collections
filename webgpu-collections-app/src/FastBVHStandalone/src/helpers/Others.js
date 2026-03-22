import { degreesToRads, radsToDegrees } from './MathUtils.js';

//================================//
export function getInfoElement()
{
    return document.getElementById("info");
}

//================================//
export function getUtilElement()
{
    return document.getElementById("utils");
}

//================================//
let profiler = null;

export function setProfilerInstance(instance)
{
    profiler = instance;
}

//================================//
export function addProfilerFrameTime(time)
{
    profiler?.addFrame(time);
}

//================================//
export function cleanUtilElement()
{
    const utilElement = getUtilElement();
    if (utilElement) {
        while (utilElement.firstChild) {
            utilElement.removeChild(utilElement.firstChild);
        }
    }
}

//================================//
export function addCheckbox(label, checkValue, utilElement, onChange)
{
    const labelElement = document.createElement('label');
    labelElement.textContent = label;
    labelElement.htmlFor = `checkbox-${label}`;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `checkbox-${label}`;
    checkbox.checked = checkValue;
    checkbox.tabIndex = -1;
    checkbox.style.cssText = `margin-left: 8px; transform: scale(1.2); cursor: pointer;`;
    checkbox.addEventListener('change', () => { onChange(checkbox.checked); });

    utilElement.appendChild(labelElement);
    utilElement.appendChild(checkbox);
    return checkbox;
}

//================================//
export function addNumberInput(label, value, min, max, step, utilElement, onChange)
{
    const labelElement = document.createElement('label');
    labelElement.textContent = label;
    labelElement.htmlFor = `number-${label}`;

    const numberInput = document.createElement('input');
    numberInput.type = 'number';
    numberInput.id = `number-${label}`;
    numberInput.value = value.toString();
    numberInput.min = min.toString();
    numberInput.max = max.toString();
    numberInput.step = step.toString();
    numberInput.tabIndex = -1;
    numberInput.style.cssText = `margin-left: 16px; transform: scale(1.2); cursor: pointer;`;
    numberInput.addEventListener('change', () => {
        const val = parseFloat(numberInput.value);
        onChange(isNaN(val) ? 0 : val);
    });

    utilElement.appendChild(labelElement);
    utilElement.appendChild(numberInput);
    return numberInput;
}

//================================//
export function addSlider(label, value, min, max, step, utilElement, onChange)
{
    const labelElement = document.createElement('label');
    labelElement.textContent = `${label}: ${value.toFixed(2)}`;
    labelElement.htmlFor = `slider-${label}`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `slider-${label}`;
    slider.min = min.toString();
    slider.max = max.toString();
    slider.step = step.toString();
    slider.value = value.toString();
    slider.style.cssText = `width: 150px; margin-left: 8px; cursor: pointer;`;
    slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        onChange(isNaN(val) ? 0 : val);
        labelElement.textContent = `${label}: ${val.toFixed(2)}`;
    });

    utilElement.appendChild(labelElement);
    utilElement.appendChild(slider);
    return slider;
}

//================================//
export function addButton(label, utilElement, onClick)
{
    const button = document.createElement('button');
    button.style.cssText = 'background-color: #444444; color: white; border: none; padding: 5px 10px; margin-top: 5px; cursor: pointer;';
    button.textContent = label;
    button.tabIndex = -1;
    button.addEventListener('click', onClick);
    utilElement.appendChild(button);
    return button;
}

//================================//
export function createMaterialContextMenu(position, currentMaterial, onApplyCallback, onCancelCallback)
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
    title.style.cssText = `font-weight: bold; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #555;`;
    menu.appendChild(title);

    // Albedo
    const albedoSection = document.createElement('div');
    albedoSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';

    const albedoLabel = document.createElement('label');
    albedoLabel.textContent = 'Albedo:';
    albedoSection.appendChild(albedoLabel);

    const toHex = (val) => Math.round(val * 255).toString(16).padStart(2, '0');
    const currentHex = `#${toHex(currentMaterial.albedo[0])}${toHex(currentMaterial.albedo[1])}${toHex(currentMaterial.albedo[2])}`;

    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = currentHex;
    colorPicker.style.cssText = `width: 50px; height: 30px; border: none; border-radius: 4px; cursor: pointer; padding: 0;`;
    colorPicker.tabIndex = -1;
    albedoSection.appendChild(colorPicker);

    const hexDisplay = document.createElement('span');
    hexDisplay.textContent = currentHex.toUpperCase();
    hexDisplay.style.cssText = 'font-family: monospace; color: #aaa;';
    albedoSection.appendChild(hexDisplay);

    colorPicker.addEventListener('input', () => {
        hexDisplay.textContent = colorPicker.value.toUpperCase();
        currentMaterial.albedo = [
            parseInt(colorPicker.value.slice(1, 3), 16) / 255,
            parseInt(colorPicker.value.slice(3, 5), 16) / 255,
            parseInt(colorPicker.value.slice(5, 7), 16) / 255
        ];
        onApplyCallback(currentMaterial);
    });
    menu.appendChild(albedoSection);

    // Metalness
    const metalnessSection = document.createElement('div');
    metalnessSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const metalnessLabel = document.createElement('label');
    metalnessLabel.textContent = `Metalness: ${currentMaterial.metalness.toFixed(2)}`;
    metalnessSection.appendChild(metalnessLabel);
    const metalnessSlider = document.createElement('input');
    metalnessSlider.type = 'range'; metalnessSlider.min = '0'; metalnessSlider.max = '1'; metalnessSlider.step = '0.01';
    metalnessSlider.value = currentMaterial.metalness.toString();
    metalnessSlider.style.cssText = 'flex: 1; cursor: pointer;';
    metalnessSlider.tabIndex = -1;
    metalnessSection.appendChild(metalnessSlider);
    metalnessSlider.addEventListener('input', () => {
        const val = parseFloat(metalnessSlider.value);
        currentMaterial.metalness = isNaN(val) ? 0 : val;
        metalnessLabel.textContent = `Metalness: ${currentMaterial.metalness.toFixed(2)}`;
        onApplyCallback(currentMaterial);
    });
    menu.appendChild(metalnessSection);

    // Roughness
    const roughnessSection = document.createElement('div');
    roughnessSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const roughnessLabel = document.createElement('label');
    roughnessLabel.textContent = `Roughness: ${currentMaterial.roughness.toFixed(2)}`;
    roughnessSection.appendChild(roughnessLabel);
    const roughnessSlider = document.createElement('input');
    roughnessSlider.type = 'range'; roughnessSlider.min = '0'; roughnessSlider.max = '1'; roughnessSlider.step = '0.01';
    roughnessSlider.value = currentMaterial.roughness.toString();
    roughnessSlider.style.cssText = 'flex: 1; cursor: pointer;';
    roughnessSlider.tabIndex = -1;
    roughnessSection.appendChild(roughnessSlider);
    roughnessSlider.addEventListener('input', () => {
        const val = parseFloat(roughnessSlider.value);
        currentMaterial.roughness = isNaN(val) ? 0 : val;
        roughnessLabel.textContent = `Roughness: ${currentMaterial.roughness.toFixed(2)}`;
        onApplyCallback(currentMaterial);
    });
    menu.appendChild(roughnessSection);

    // Buttons
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';
    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `padding: 6px 16px; background: #555; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 13px;`;
    cancelButton.tabIndex = -1;
    cancelButton.addEventListener('click', () => { onCancelCallback(); });
    buttonsRow.appendChild(cancelButton);
    menu.appendChild(buttonsRow);

    return menu;
}

//================================//
export function createLightContextMenu(position, currentLight, title, onApplyCallback, onCancelCallback)
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
    titlediv.style.cssText = `font-weight: bold; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #555;`;
    menu.appendChild(titlediv);

    // Enabled
    const enabledSection = document.createElement('div');
    enabledSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const enabledLabel = document.createElement('label');
    enabledLabel.textContent = 'Enabled:';
    enabledSection.appendChild(enabledLabel);
    const enabledCheckbox = document.createElement('input');
    enabledCheckbox.type = 'checkbox';
    enabledCheckbox.checked = currentLight.enabled;
    enabledCheckbox.tabIndex = -1;
    enabledSection.appendChild(enabledCheckbox);
    enabledCheckbox.addEventListener('change', () => {
        currentLight.enabled = enabledCheckbox.checked;
        onApplyCallback(currentLight);
    });
    menu.appendChild(enabledSection);

    // Position XYZ
    const positionSection = document.createElement('div');
    positionSection.style.cssText = 'display: flex; flex-direction: row; gap: 6px; margin-bottom: 12px;';
    const positionLabel = document.createElement('label');
    positionLabel.textContent = 'Light position:';
    positionSection.appendChild(positionLabel);
    ['X', 'Y', 'Z'].forEach((axis, index) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.value = currentLight.position[index].toFixed(2);
        input.placeholder = axis;
        input.style.cssText = `width: 50px; padding: 4px; border: 1px solid #555; border-radius: 4px; background: #222; color: #eee;`;
        input.tabIndex = -1;
        positionSection.appendChild(input);
        input.addEventListener('input', () => {
            const val = parseFloat(input.value);
            currentLight.position[index] = isNaN(val) ? 0 : val;
            onApplyCallback(currentLight);
        });
    });
    menu.appendChild(positionSection);

    // Direction XYZ
    const directionSection = document.createElement('div');
    directionSection.style.cssText = 'display: flex; flex-direction: row; gap: 6px; margin-bottom: 12px;';
    const directionLabel = document.createElement('label');
    directionLabel.textContent = 'Light direction:';
    directionSection.appendChild(directionLabel);
    ['X', 'Y', 'Z'].forEach((axis, index) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.value = currentLight.direction[index].toFixed(2);
        input.placeholder = axis;
        input.style.cssText = `width: 50px; padding: 4px; border: 1px solid #555; border-radius: 4px; background: #222; color: #eee;`;
        input.tabIndex = -1;
        directionSection.appendChild(input);
        input.addEventListener('input', () => {
            const val = parseFloat(input.value);
            currentLight.direction[index] = isNaN(val) ? 0 : val;
            onApplyCallback(currentLight);
        });
    });
    menu.appendChild(directionSection);

    // Intensity
    const intensitySection = document.createElement('div');
    intensitySection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const intensityLabel = document.createElement('label');
    intensityLabel.textContent = 'Light intensity:';
    intensitySection.appendChild(intensityLabel);
    const intensityInput = document.createElement('input');
    intensityInput.type = 'number';
    intensityInput.value = currentLight.intensity.toFixed(2);
    intensityInput.style.cssText = `width: 80px; padding: 4px; border: 1px solid #555; border-radius: 4px; background: #222; color: #eee;`;
    intensityInput.tabIndex = -1;
    intensitySection.appendChild(intensityInput);
    intensityInput.addEventListener('input', () => {
        const val = parseFloat(intensityInput.value);
        currentLight.intensity = isNaN(val) ? 0 : val;
        onApplyCallback(currentLight);
    });
    menu.appendChild(intensitySection);

    // Cone angle
    const coneSection = document.createElement('div');
    coneSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const coneLabel = document.createElement('label');
    coneLabel.textContent = 'Cone angle:';
    coneSection.appendChild(coneLabel);
    const coneInput = document.createElement('input');
    coneInput.type = 'range';
    coneInput.value = radsToDegrees(currentLight.coneAngle).toFixed(2);
    coneInput.min = '0'; coneInput.max = '180';
    coneInput.style.cssText = `width: 80px; cursor: pointer;`;
    coneInput.tabIndex = -1;
    coneSection.appendChild(coneInput);
    coneInput.addEventListener('input', () => {
        const val = parseFloat(coneInput.value);
        currentLight.coneAngle = isNaN(val) ? 0 : degreesToRads(val);
        onApplyCallback(currentLight);
    });
    menu.appendChild(coneSection);

    // Color
    const lightColorSection = document.createElement('div');
    lightColorSection.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 12px;';
    const colorLabel = document.createElement('label');
    colorLabel.textContent = 'Light color:';
    lightColorSection.appendChild(colorLabel);

    const toHex = (val) => Math.round(val * 255).toString(16).padStart(2, '0');
    const currentHex = `#${toHex(currentLight.color[0])}${toHex(currentLight.color[1])}${toHex(currentLight.color[2])}`;

    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = currentHex;
    colorPicker.style.cssText = `width: 50px; height: 30px; border: none; border-radius: 4px; cursor: pointer; padding: 0;`;
    colorPicker.tabIndex = -1;
    lightColorSection.appendChild(colorPicker);

    const hexDisplay = document.createElement('span');
    hexDisplay.textContent = currentHex.toUpperCase();
    hexDisplay.style.cssText = 'font-family: monospace; color: #aaa;';
    lightColorSection.appendChild(hexDisplay);

    colorPicker.addEventListener('input', () => {
        hexDisplay.textContent = colorPicker.value.toUpperCase();
        currentLight.color = [
            parseInt(colorPicker.value.slice(1, 3), 16) / 255,
            parseInt(colorPicker.value.slice(3, 5), 16) / 255,
            parseInt(colorPicker.value.slice(5, 7), 16) / 255
        ];
        onApplyCallback(currentLight);
    });
    menu.appendChild(lightColorSection);

    // Buttons
    const buttonsRow = document.createElement('div');
    buttonsRow.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end;';
    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `padding: 6px 16px; background: #555; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 13px;`;
    cancelButton.tabIndex = -1;
    cancelButton.addEventListener('click', () => { onCancelCallback(); });
    buttonsRow.appendChild(cancelButton);
    menu.appendChild(buttonsRow);

    return menu;
}
