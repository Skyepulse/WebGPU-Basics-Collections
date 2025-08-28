<template>
    <div class="flex justify-center items-center w-full h-full">
        <canvas id="webgpuCanvas" ref="webgpuCanvas" class="w-[40%] aspect-square"></canvas>
    </div>
</template>

<script setup lang="ts">
    import { onMounted, ref } from 'vue';
    import { startup_1 } from './1-BasicStart/main';
    import { startup_2 } from './2-ComputeBasics/main';
    import { startup_3 } from './3-VariablesAndUniforms/main';

    const webgpuCanvas = ref<HTMLCanvasElement | null>(null);

    //================================//
    onMounted(() => {
        if (webgpuCanvas.value) {
            startup_3(webgpuCanvas.value);
        }
    });
</script>
