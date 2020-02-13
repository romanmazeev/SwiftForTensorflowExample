//
//  TestImageLoader.swift
//  SwiftForTensorflowExample
//
//  Created by Roman Mazeev on 05.01.2020.
//  Copyright © 2020 Roman Mazeev. All rights reserved.
//

import Cocoa
import TensorFlow

class TestImageLoader {
    func readDigit(filePath: String) -> Tensor<Float> {
        guard let image = NSImage(byReferencingFile: filePath) else { fatalError("Can`t load test image") }
        var proposedRect = CGRect(x: 0, y: 0, width: 28, height: 28)
        guard let pixelValues = image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil)?.pixelValues else {
            fatalError("Cant convert image to CGImage")
        }

        let scalars = pixelValues.map { Float(Int($0)) }

        return Tensor(shape: [1, 1, Int(image.size.height), Int(image.size.width)], scalars: scalars)
            .transposed(permutation: [0, 2, 3, 1]) / 255 // NHWC
    }
}

extension CGImage {
    var pixelValues: [UInt8] {
        var pixelValues = [UInt8](repeating: 0, count: height * bytesPerRow)

        let contextRef = CGContext(data: &pixelValues,
                                   width: width,
                                   height: height,
                                   bitsPerComponent: bitsPerComponent,
                                   bytesPerRow: bytesPerRow,
                                   space: CGColorSpaceCreateDeviceGray(),
                                   bitmapInfo: 0)
        contextRef?.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return pixelValues
    }
}
