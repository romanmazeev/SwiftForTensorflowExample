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
    func getDigit() -> Tensor<Float> {
        guard let imageURL = selectFile() else { fatalError("Cant get image") }
        let image = NSImage(byReferencing: imageURL)
        var proposedRect = CGRect(x: 0, y: 0, width: 28, height: 28)
        guard let pixelValues = image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil)?.pixelScalarValues else {
            fatalError("Cant convert image to CGImage")
        }

        return Tensor(shape: [1, 1, Int(image.size.height), Int(image.size.width)], scalars: pixelValues)
            .transposed(permutation: [0, 2, 3, 1]) / 255 // NHWC (Number of samples x Height x Width x Channels)
    }

    func selectFile() -> URL? {
        let dialog = NSOpenPanel()
        dialog.allowedFileTypes = ["jpg"]
        dialog.allowsMultipleSelection = false
        dialog.canChooseDirectories = false
        guard dialog.runModal() == .OK else { return nil }
        return dialog.url
    }
}

extension CGImage {
    var pixelScalarValues: [Float] {
        var pixelValues = [UInt8](repeating: 0, count: height * bytesPerRow)

        let contextRef = CGContext(data: &pixelValues,
                                   width: width,
                                   height: height,
                                   bitsPerComponent: bitsPerComponent,
                                   bytesPerRow: bytesPerRow,
                                   space: CGColorSpaceCreateDeviceGray(),
                                   bitmapInfo: 0)
        contextRef?.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return pixelValues.map { Float(Int($0)) }
    }
}
