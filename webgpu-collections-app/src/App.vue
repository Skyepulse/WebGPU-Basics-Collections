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
                @click="() => selectExample(i)"
                @mouseenter="onButtonHover(i, $event)"
                @mouseleave="onButtonLeave"
            >
                {{ i }}
            </button>
        </div>
        <canvas id="webgpuCanvas" ref="webgpuCanvas" class="w-[90%] h-full"></canvas>
        
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

    const webgpuCanvas = ref<HTMLCanvasElement | null>(null);

    const numberOfExamples = 4;
    const startupFunctions = [startup_1, startup_2, startup_3, startup_4];
    const startupNames = ['Basic Start', 'Compute Basics', 'Variables and Uniforms', 'Storage Buffer Instancing'];

    // Slider state
    const hoveredIndex = ref<number|null>(null);
    const hoveredButtonTop = ref(0);
    const hoveredButtonHeight = ref(0);

    function selectExample(index: number) {
        if (webgpuCanvas.value) {
            const fn = startupFunctions[index - 1];
            if (fn) fn(webgpuCanvas.value);
        }
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
        selectExample(1);
    });
</script>
