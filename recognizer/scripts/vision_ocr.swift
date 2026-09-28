// OCR-only local worker; separate from the immutable historical media-audit worker.
import Foundation
import Vision
import ImageIO

func request() -> VNRecognizeTextRequest {
    let value = VNRecognizeTextRequest()
    value.revision = VNRecognizeTextRequestRevision3
    value.recognitionLevel = .accurate
    value.recognitionLanguages = ["ru-RU", "en-US"]
    value.usesLanguageCorrection = false
    return value
}

func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
       let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}

if CommandLine.arguments.contains("--info") {
    let ocr = request()
    emit(["revision": ocr.revision, "languages": ocr.recognitionLanguages,
          "supported_languages": (try? ocr.supportedRecognitionLanguages()) ?? [],
          "os": ProcessInfo.processInfo.operatingSystemVersionString,
          "recognition_level": "accurate", "language_correction": false])
} else {
    while let line = readLine() {
        autoreleasepool {
            var result: [String: Any] = [:]
            do {
                guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: String],
                      let path = row["path"], let id = row["id"] else {
                    throw NSError(domain: "VisionOCR", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "Expected path and id"])
                }
                result["id"] = id
                let ocr = request()
                try VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:]).perform([ocr])
                result["observations"] = (ocr.results ?? []).compactMap { observation -> [String: Any]? in
                    guard let candidate = observation.topCandidates(1).first else { return nil }
                    let b = observation.boundingBox
                    return ["raw_text": candidate.string, "score": candidate.confidence,
                            "bbox_bottom_left_normalized": [b.minX, b.minY, b.maxX, b.maxY]]
                }
                result["revision"] = ocr.revision
                result["status"] = "ok"
            } catch {
                result["status"] = "error"
                result["error"] = String(describing: error)
            }
            emit(result)
        }
    }
}
