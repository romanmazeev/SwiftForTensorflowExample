//
//  DatasetUtilities.swift
//  SwiftForTensorflowExample
//
//  Created by Roman Mazeev on 05.01.2020.
//  Copyright © 2020 Roman Mazeev. All rights reserved.
//

import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
 
struct DatasetUtilities {
    struct ResourceDefinition {
        let filename: String
        let remoteRoot: URL
        let localStorageDirectory: URL

        var localURL: URL { localStorageDirectory.appendingPathComponent(filename) }
        var remoteURL: URL { remoteRoot.appendingPathComponent(filename).appendingPathExtension("gz") }
        var archiveURL: URL { localURL.appendingPathExtension("gz") }
    }

    static let currentWorkingDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    static func loadTestImage(filename: String, localStorageDirectory: URL = currentWorkingDirectoryURL) -> Data {
        print("Loading resource: \(filename)")
        let localURL = localStorageDirectory.appendingPathComponent(filename)

        do {
            print("Loading local data at: \(localURL.path)")
            let data = try Data(contentsOf: localURL)
            print("Succesfully loaded resource: \(filename)")
            return data
        } catch {
            fatalError("Failed to contents of resource: \(localURL)")
        }
    }

    public static func fetchResource(filename: String, remoteRoot: URL, localStorageDirectory: URL = currentWorkingDirectoryURL) -> Data {
        print("Loading resource: \(filename)")

        let resource = ResourceDefinition(
            filename: filename,
            remoteRoot: remoteRoot,
            localStorageDirectory: localStorageDirectory)

        let localURL = resource.localURL

        if !FileManager.default.fileExists(atPath: localURL.path) {
            print(
                "File does not exist locally at expected path: \(localURL.path) and must be fetched"
            )
            fetchFromRemoteAndSave(resource)
        }

        do {
            print("Loading local data at: \(localURL.path)")
            let data = try Data(contentsOf: localURL)
            print("Succesfully loaded resource: \(filename)")
            return data
        } catch {
            fatalError("Failed to contents of resource: \(localURL)")
        }
    }

    static func fetchFromRemoteAndSave(_ resource: ResourceDefinition) {
        let remoteLocation = resource.remoteURL
        let archiveLocation = resource.archiveURL

        do {
            print("Fetching URL: \(remoteLocation)...")
            let archiveData = try Data(contentsOf: remoteLocation)
            print("Writing fetched archive to: \(archiveLocation.path)")
            try archiveData.write(to: archiveLocation)
        } catch {
            fatalError("Failed to fetch and save resource with error: \(error)")
        }
        print("Archive saved to: \(archiveLocation.path)")

        extractArchive(for: resource)
    }

    static func extractArchive(for resource: ResourceDefinition) {
        print("Extracting archive...")

        let archivePath = resource.archiveURL.path

        #if os(macOS)
            let gunzipLocation = "/usr/bin/gunzip"
        #else
            let gunzipLocation = "/bin/gunzip"
        #endif

        let task = Process()
        task.executableURL = URL(fileURLWithPath: gunzipLocation)
        task.arguments = [archivePath]
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            fatalError("Failed to extract \(archivePath) with error: \(error)")
        }
    }
}
