//
//  ImageClassificationDataset.swift
//  TensorFlowSwiftProj
//
//  Created by Roman Mazeev on 05.01.2020.
//  Copyright © 2020 Roman Mazeev. All rights reserved.
//

import TensorFlow

protocol ImageClassificationDataset {
    init()
    var trainingDataset: Dataset<LabeledExample> { get }
    var testDataset: Dataset<LabeledExample> { get }
    var trainingExampleCount: Int { get }
    var testExampleCount: Int { get }
}
