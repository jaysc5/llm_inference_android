package com.google.mediapipe.examples.llminference

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.io.File
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

class InferenceModel private constructor(context: Context) {
    private var llmInference: LlmInference

    private var startTime: Long = 0L
    private var firstTokenTime: Long = 0L
    private var lastTokenTime: Long = 0L
    private var tokenCount: Int = 0

    private val modelExists: Boolean
        get() = File(MODEL_PATH).exists()

    private val _partialResults = MutableSharedFlow<Pair<String, Boolean>>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val partialResults: SharedFlow<Pair<String, Boolean>> = _partialResults.asSharedFlow()

    init {
        if (!modelExists) {
            throw IllegalArgumentException("Model not found at path: $MODEL_PATH")
        }

        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(MODEL_PATH)
            .setMaxTokens(1024)
            .setResultListener { partialResult, done ->
                val currentTime = System.currentTimeMillis()
                if (tokenCount == 0) firstTokenTime = currentTime - startTime
                lastTokenTime = currentTime
                tokenCount++

                _partialResults.tryEmit(partialResult to done)
            }
            .build()

        llmInference = LlmInference.createFromOptions(context, options)
    }

    fun generateResponseAsync(prompt: String) {
        startTime = System.currentTimeMillis()
        firstTokenTime = 0L
        lastTokenTime = 0L
        tokenCount = 0

        // Add the gemma prompt prefix to trigger the response.
        val gemmaPrompt = prompt + "<start_of_turn>model\n"
        llmInference.generateResponseAsync(gemmaPrompt)
    }

    fun getMetrics(): String {
        val latency = lastTokenTime - startTime
        val throughput = if (latency > 0) tokenCount * 1000.0 / latency else 0.0
        val tpot = if (tokenCount > 1) (latency - firstTokenTime) / (tokenCount - 1) else 0L

        return "\nTTFT: ${firstTokenTime}ms\nTPOT: ${tpot}ms\nLatency: ${latency}ms\nThroughput: ${throughput} tokens/sec"
    }

    companion object {
        // NB: Make sure the filename is *unique* per model you use!
        // Weight caching is currently based on filename alone.
        private const val MODEL_PATH = "/data/local/tmp/llm/gemma2-2b-it-gpu-int8.bin"
        private var instance: InferenceModel? = null

        fun getInstance(context: Context): InferenceModel {
            return if (instance != null) {
                instance!!
            } else {
                InferenceModel(context).also { instance = it }
            }
        }
    }
}
