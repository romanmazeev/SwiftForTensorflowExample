//
//  MNIST.swift
//  SwiftForTensorflowExample
//
//  Created by Roman Mazeev on 05.01.2020.
//  Copyright © 2020 Roman Mazeev. All rights reserved.
//

import Foundation
import TensorFlow

public struct MNIST: ImageClassificationDataset {
    let trainingDataset: Dataset<LabeledExample>
    let testDataset: Dataset<LabeledExample>
    let trainingExampleCount = 60000
    let testExampleCount = 10000

    init() {
        self.init(flattening: false, normalizing: false)
    }

    init(flattening: Bool = false,
                normalizing: Bool = false,
                localStorageDirectory: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNIST")) {
        self.trainingDataset = Dataset<LabeledExample>(elements: fetchDataset(localStorageDirectory: localStorageDirectory,
                                                                              imagesFilename: "train-images-idx3-ubyte",
                                                                              labelsFilename: "train-labels-idx1-ubyte",
                                                                              flattening: flattening,
                                                                              normalizing: normalizing))

        self.testDataset = Dataset<LabeledExample>(elements: fetchDataset(localStorageDirectory: localStorageDirectory,
                                                                          imagesFilename: "t10k-images-idx3-ubyte",
                                                                          labelsFilename: "t10k-labels-idx1-ubyte",
                                                                          flattening: flattening,
                                                                          normalizing: normalizing))
    }
}
 func fetchDataset(localStorageDirectory: URL,
                              imagesFilename: String,
                              labelsFilename: String,
                              flattening: Bool,
                              normalizing: Bool) -> LabeledExample {
    guard let remoteRoot = URL(string: "http://yann.lecun.com/exdb/mnist") else {
        fatalError("Failed to create MNIST root url: http://yann.lecun.com/exdb/mnist")
    }

    if !FileManager.default.fileExists(atPath: localStorageDirectory.path) {
        do {
            try FileManager.default.createDirectory(
                at: localStorageDirectory, withIntermediateDirectories: false)
        } catch {
            fatalError(
                "Failed to create storage directory: \(localStorageDirectory.path), error: \(error)"
            )
        }
    }

    let imagesData = DatasetUtilities.fetchResource(filename: imagesFilename,
                                                    remoteRoot: remoteRoot,
                                                    localStorageDirectory: localStorageDirectory)
    let labelsData = DatasetUtilities.fetchResource(filename: labelsFilename,
                                                    remoteRoot: remoteRoot,
                                                    localStorageDirectory: localStorageDirectory)

    let images = [UInt8](imagesData).dropFirst(16).map(Float.init)
    let labels = [UInt8](labelsData).dropFirst(8).map(Int32.init)

    let rowCount = labels.count
    let (imageWidth, imageHeight) = (28, 28)

    if flattening {
        var flattenedImages =
            Tensor(shape: [rowCount, imageHeight * imageWidth], scalars: images)
            / 255.0
        if normalizing {
            flattenedImages = flattenedImages * 2.0 - 1.0
        }
        return LabeledExample(label: Tensor(labels), data: flattenedImages)
    } else {
        return LabeledExample(
            label: Tensor(labels),
            data:
                Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
                .transposed(permutation: [0, 2, 3, 1]) / 255  // NHWC (Number of samples x Height x Width x Channels)
        )
    }
}

