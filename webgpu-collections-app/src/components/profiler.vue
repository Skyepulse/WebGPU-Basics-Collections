<template>
  <div class="flex flex-row items-stretch h-20 font-mono text-[10px] select-none text-[#aaa]">

    <!-- chart -->
    <div class="flex-1 overflow-hidden relative bg-[#1a1a1a]/20 rounded-sm">
      <div
        class="absolute left-0 bottom-0 top-0 flex flex-row items-end"
        :style="{ width: (frames.length / props.maxBars) * 100 + '%' }"
      >
        <div
          v-for="frame in frames"
          :key="frame.id"
          class="flex-1 min-w-0 rounded-t-[1px]"
          :title="`${frame.duration.toFixed(1)}ms`"
          :style="{ height: barHeightPct(frame.duration) + '%', backgroundColor: barColor(frame.duration) }"
        >
        </div>
      </div>
    </div>

    <!-- scale -->
    <div class="w-[30px] relative shrink-0 ml-[3px]">
      <div class="absolute left-0 top-0 bottom-0 w-px bg-[#555]"></div>
      <span class="absolute left-1 top-0 leading-none whitespace-nowrap">{{ scaleMax }}</span>
      <span class="absolute left-1 top-1/2 -translate-y-1/2 leading-none whitespace-nowrap">{{ scaleMid }}</span>
      <span class="absolute left-1 bottom-0 leading-none whitespace-nowrap">0</span>
    </div>
  </div>
  
</template>

<script setup lang="ts">
    import { ref, computed } from 'vue';

    //================================//
    interface FrameEntry
    {
        id: number,
        duration: number
    }

    //================================//
    const props = withDefaults(defineProps<{
            maxBars?: number
        }>(),
        {
            maxBars: 60
    });

    //================================//
    let nextId = 0;
    const frames = ref<FrameEntry[]>([]);
    const dynamicMax = ref(60);

    //================================//
    const scaleMax = computed(() => dynamicMax.value);
    const scaleMid = computed(() => Math.round(dynamicMax.value / 2));

    //================================//
    function barHeightPct(duration: number): number
    {
        return Math.min((duration / dynamicMax.value) * 100, 100);
    }

    //================================//
    function barColor(duration: number): string
    {
        const t = Math.min(duration / 60, 1);
        const r = Math.round(255 * (1 - t));
        const g = Math.round(255 * t);
        return `rgb(${r},${g},0)`;
    }

    //================================//
    function addFrame(duration: number)
    {

        if (frames.value.length >= props.maxBars)
            frames.value.shift();

        frames.value.push({ id: nextId++, duration: Math.min(Math.max(duration, 0), 60)});
    }

    // API CALL POINT
    defineExpose({ addFrame });
</script>
