// Signalsmith Stretch の薄いC ABIラッパー（モノラル・ストリーミング用）。
//
// PyPIの python-stretch はワンショットAPI（呼び出しごとに seek/flush）のため
// 可変レートストレッチ・時間変化ピッチシフトに使えない。本ラッパーは
// SignalsmithStretch のストリーミング process/flush をそのまま公開する。
//
// ベンダリングしたヘッダ（vendor/）は MIT License:
//   signalsmith-stretch  (c) Geraint Luff / Signalsmith Audio Ltd.
//   signalsmith-linear   (c) Signalsmith Audio

#include "vendor/signalsmith-stretch.h"

// MSVCは extern "C" だけではDLLエクスポートされない
#ifdef _WIN32
#define VS_EXPORT __declspec(dllexport)
#else
#define VS_EXPORT
#endif

using Stretch = signalsmith::stretch::SignalsmithStretch<float>;

extern "C" {

VS_EXPORT void *vs_create(float sample_rate) {
    auto *s = new Stretch();
    s->presetDefault(1, sample_rate);
    return s;
}

VS_EXPORT void vs_destroy(void *p) {
    delete static_cast<Stretch *>(p);
}

VS_EXPORT void vs_reset(void *p) {
    static_cast<Stretch *>(p)->reset();
}

VS_EXPORT int vs_input_latency(void *p) {
    return static_cast<Stretch *>(p)->inputLatency();
}

VS_EXPORT int vs_output_latency(void *p) {
    return static_cast<Stretch *>(p)->outputLatency();
}

VS_EXPORT void vs_set_transpose_factor(void *p, float multiplier, float tonality_limit) {
    static_cast<Stretch *>(p)->setTransposeFactor(multiplier, tonality_limit);
}

VS_EXPORT void vs_set_formant_factor(void *p, float multiplier, int compensate) {
    static_cast<Stretch *>(p)->setFormantFactor(multiplier, compensate != 0);
}

VS_EXPORT void vs_process(void *p, const float *in, int in_samples,
                float *out, int out_samples) {
    const float *ins[1] = {in};
    float *outs[1] = {out};
    static_cast<Stretch *>(p)->process(ins, in_samples, outs, out_samples);
}

VS_EXPORT void vs_flush(void *p, float *out, int out_samples) {
    float *outs[1] = {out};
    static_cast<Stretch *>(p)->flush(outs, out_samples);
}

}  // extern "C"
