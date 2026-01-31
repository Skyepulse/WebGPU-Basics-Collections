<style scoped>
#indexingContainer {
  direction: rtl;
}
#indexingContainer > button {
  direction: ltr;
}
</style>

<template>
    <div class="flex justify-center items-center w-full h-full">
        <div
            id="indexingContainer"
            class="w-[10%] h-full bg-gray-800 flex flex-col justify-start items-center py-1 overflow-y-auto"
        >
            <button
                v-for="i in numberOfExamples"
                :key="i"
                class="w-full h-20 last:mb-0 border border-gray-300 hover:bg-amber-300 active:bg-amber-500 text-lg font-bold shadow flex-shrink-0 relative"
                tabindex="-1"
                @click="() => selectExample(i)"
                @keydown.space.prevent
                @keydown.enter.prevent
                @mouseenter="onButtonHover(i, $event)"
                @mouseleave="onButtonLeave"
            >
                {{ i }}
            </button>
        </div>
        <canvas id="webgpuCanvas" ref="webgpuCanvas" class="w-[90%] h-full"></canvas>
        <pre id="info" class="absolute top-0 right-0 p-4"></pre>
        <pre id="utils" class="absolute bottom-0 right-0 p-1 bg-gray-700"></pre>

        <!-- Animated slider (width grows left->right via scaleX) -->
        <div
            class="absolute left-[10%] w-[25%] bg-gray-700 text-white flex items-center justify-center font-bold text-lg pointer-events-none select-none shadow-lg origin-left transition-all duration-200"
            :class="hoveredIndex === null ? 'opacity-0 scale-x-0' : 'opacity-100 scale-x-100'"
            :style="sliderStyle"
        >
                {{ hoveredName }}
        </div>
    </div>
</template>

<script setup lang="ts">
    import { onMounted, ref, computed } from 'vue';
    import { startup_1 } from './1-BasicStart/main';
    import { startup_2 } from './2-ComputeBasics/main';
    import { startup_3 } from './3-VariablesAndUniforms/main';
    import { startup_4 } from './4-StorageBufferInstancing/main';
    import { startup_5 } from './5-VertexAndIndexBuffers/main';
    import { startup_6 } from './6-Textures/main';
    import { startup_7 } from './7-Game/main';
    import { startup_8 } from './8-RayTrace/main';
    import { startup_9 } from './9-Transparency/main';
    import { startup_10 } from './10-PBR/main';

    const webgpuCanvas = ref<HTMLCanvasElement | null>(null);
    const currentRenderer = ref<any>(null);
    const isSwitching = ref(false);

    const startupFunctions = [startup_1, startup_2, startup_3, startup_4, startup_5, startup_6, startup_7, startup_8, startup_9, startup_10];
    const numberOfExamples = startupFunctions.length;
    const startupNames = ['Basic Start', 'Compute Basics', 'Variables and Uniforms', 'Storage Buffer Instancing', 'Vertex and Index Buffers', 'Textures', 'Game', 'Ray Trace', 'Transparency', 'PBR'];

    // Slider state
    const hoveredIndex = ref<number|null>(null);
    const hoveredButtonTop = ref(0);
    const hoveredButtonHeight = ref(0);

    async function selectExample(index: number) {
        if (isSwitching.value) return; // Prevent multiple clicks while switching
        isSwitching.value = true;

        if (currentRenderer.value && typeof currentRenderer.value.cleanup === 'function') {
            await currentRenderer.value.cleanup();
            currentRenderer.value = null;
        }

        if (webgpuCanvas.value) {
            const fn = startupFunctions[index - 1];
            if (fn) currentRenderer.value = await fn(webgpuCanvas.value);
        }

        isSwitching.value = false;
    }

    //================================//
    function onButtonHover(i: number, event: MouseEvent) {
        hoveredIndex.value = i;
        const target = event.currentTarget as HTMLElement;

        // Get button position relative to parent (indexingContainer)
        const parent = target.parentElement;
        if (parent) {
            const parentRect = parent.getBoundingClientRect();
            const btnRect = target.getBoundingClientRect();
            hoveredButtonTop.value = btnRect.top - parentRect.top;
            hoveredButtonHeight.value = btnRect.height;
        }
    }

    //================================//
    function onButtonLeave() {
        hoveredIndex.value = null;
    }

    //================================//
    const hoveredName = computed(() =>
        hoveredIndex.value !== null ? startupNames[hoveredIndex.value - 1] : ''
    );

    //================================//
    const sliderStyle = computed(() => {
        if (hoveredIndex.value === null) {
            return {
            top: hoveredButtonTop.value + 'px',
            height: hoveredButtonHeight.value + 'px',
            transition: 'top 0.2s cubic-bezier(0.4,0,0.2,1), height 0.2s cubic-bezier(0.4,0,0.2,1)'
            };
        }
        return {
            top: hoveredButtonTop.value + 'px',
            height: hoveredButtonHeight.value + 'px',
            transition: 'top 0.2s cubic-bezier(0.4,0,0.2,1), height 0.2s cubic-bezier(0.4,0,0.2,1)'
        };
    });

    //================================//
    onMounted(() => {
        selectExample(8);
    });
</script>
