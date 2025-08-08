<template>
    <div class="flex justify-center items-center w-full h-full">
        <canvas id="webgpuCanvas" ref="webgpuCanvas" class="w-[50%]"></canvas>
    </div>
</template>

<script setup lang="ts">
    import { onMounted, ref } from 'vue';
    import { startup } from './1-BasicStart/main';

    const webgpuCanvas = ref<HTMLCanvasElement | null>(null);

    //================================//
    onMounted(() => {
        if (webgpuCanvas.value) {
            startup(webgpuCanvas.value);
        }
    });
</script>
