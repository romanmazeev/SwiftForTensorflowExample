//
//  main.swift
//  SwiftForTensorflowExample
//
//  Created by Roman Mazeev on 20.12.2019.
//  Copyright © 2019 Roman Mazeev. All rights reserved.
//

import TensorFlow
import Foundation
import AppKit

typealias Classifier = Sequential<
    Conv2D<Float>, Sequential<
        AvgPool2D<Float>, Sequential<
            Conv2D<Float>,Sequential<
                AvgPool2D<Float>, Sequential<
                    Flatten<Float>, Sequential<
                        Dense<Float>, Sequential<
                            Dense<Float>, Dense<Float>
                        >
                    >
                >
            >
        >
    >
>

struct Statistics {
    var correctGuessCount = 0
    var totalGuessCount = 0
    var totalLoss: Float = 0
    var batches = 0
}

private var classifier = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 84, outputSize: 10)
}
private let optimizer = SGD(for: classifier, learningRate: 0.1)

// MARK: - Training
struct Constants {
    static let epochCount = 12
    static let batchSize = 128
}

func startTraining() {
    func logTrain(epoch: Int, trainStatistics: Statistics, testStatistics: Statistics) {
        print(
            """
            [Epoch \(epoch)] \
            Training Loss : \(trainStatistics.totalLoss / Float(trainStatistics.batches)), \
            Training Accuracy: \(trainStatistics.correctGuessCount)/\(trainStatistics.totalGuessCount) \
            (\(Float(trainStatistics.correctGuessCount) / Float(trainStatistics.totalGuessCount))), \
            Test Loss: \(testStatistics.totalLoss / Float(testStatistics.batches)), \
            Test Accuracy: \(testStatistics.correctGuessCount)/\(testStatistics.totalGuessCount) \
            (\(Float(testStatistics.correctGuessCount) / Float(testStatistics.totalGuessCount)))
            """
        )
    }
    let dataset = MNIST()
    print("Beginning training...")
    for epoch in 1...Constants.epochCount {
        // Training dataset
        Context.local.learningPhase = .training
        var trainingStatistics = Statistics()
        let shuffledTrainingDataset = dataset.trainingDataset.shuffled(sampleCount: dataset.trainingExampleCount,
                                                                       randomSeed: Int64(epoch))
        shuffledTrainingDataset.batched(Constants.batchSize).forEach { example in
            optimizer.update(&classifier, along: TensorFlow.gradient(at: classifier) {
                getLoss(for: $0, example: example,statistics: &trainingStatistics)
            })
        }

        // Test dataset
        Context.local.learningPhase = .inference
        var testingStatistics = Statistics()
        dataset.testDataset.batched(Constants.batchSize).forEach {
            getLoss(for: classifier, example: $0, statistics: &testingStatistics)
        }

        logTrain(epoch: epoch, trainStatistics: trainingStatistics, testStatistics: testingStatistics)
    }

    @discardableResult
    func getLoss(for classifier: Classifier, example: LabeledExample, statistics: inout Statistics) -> Tensor<Float> {
        let (labels, images) = (example.label, example.data)
        let ŷ = classifier(images)
        let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== labels
        statistics.correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        statistics.totalGuessCount += example.data.shape[0]
        let loss = softmaxCrossEntropy(logits: ŷ, labels: labels)
        statistics.totalLoss += loss.scalarized()
        statistics.batches += 1
        return loss
    }
}



// MARK: - TestImage
private func testImage() {
    let tensorDigit = TestImageLoader().getDigit()
    let ŷ = classifier(tensorDigit)
    guard let predictedDigit = ŷ.argmax(squeezingAxis: 1).scalars.first else { fatalError("Can`t predict") }
    print("Test digit is: \(predictedDigit)")
}

// MARK: - Main
startTraining()
testImage()
