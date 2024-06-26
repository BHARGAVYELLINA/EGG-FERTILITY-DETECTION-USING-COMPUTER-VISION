import cv2
import numpy as np

def harris_corner_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    return image

def dog_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray, (5, 5), 1)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 2)
    dog = blur1 - blur2
    return dog

def anms(keypoints, num_points):
    if len(keypoints) <= num_points:
        return keypoints

    radii = np.zeros(len(keypoints))
    for i, keypoint in enumerate(keypoints):
        r_min = float('inf')
        for j, keypoint_compare in enumerate(keypoints):
            if keypoint.response < keypoint_compare.response:
                dist = np.linalg.norm(np.array(keypoint.pt) - np.array(keypoint_compare.pt))
                r_min = min(r_min, dist)
        radii[i] = r_min

    selected_indices = np.argsort(radii)[::-1][:num_points]
    return [keypoints[i] for i in selected_indices]

def mser_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(image, hulls, 1, (0, 255, 0))
    return image

def shi_tomasi_corner_detection(image):
    corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = np.intp(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image

def sift_detector(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image, keypoints, descriptors

def fast_detector(image):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
    return image, keypoints

def orb_detector(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return image, keypoints, descriptors

def brisk_detector(image):
    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return image, keypoints, descriptors

def kaze_detector(image):
    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
    return image, keypoints, descriptors

def akaze_detector(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    image = cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0))
    return image, keypoints, descriptors

def bf_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def flann_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
    if desc2.dtype != np.float32:
        desc2 = desc2.astype(np.float32)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def brute_force_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def flann_based_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
    if desc2.dtype != np.float32:
        desc2 = desc2.astype(np.float32)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matches = matcher.knnMatch(desc1, desc2, k=2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def knn_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def radius_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.radiusMatch(desc1, desc2, maxDistance=0.5)
    return matches

def ratio_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    ratio_thresh = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def cross_check_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher()
    matches1to2 = matcher.match(desc1, desc2)
    matches2to1 = matcher.match(desc2, desc1)

    good_matches = []
    for m1 in matches1to2:
        for m2 in matches2to1:
            if m1.queryIdx == m2.trainIdx and m2.queryIdx == m1.trainIdx:
                good_matches.append(m1)
                break
    return good_matches

def ransac_matcher(desc1, desc2, kp1, kp2, mask):
    if desc1 is None or desc2 is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return matchesMask

def brute_force_cross_check_matcher(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != desc2.dtype:
        desc1 = desc1.astype(desc2.dtype)

    if desc1.shape[1] != desc2.shape[1]:
        return []

    matcher = cv2.BFMatcher()
    matches1to2 = matcher.match(desc1, desc2)
    matches2to1 = matcher.match(desc2, desc1)

    good_matches = []
    for m1 in matches1to2:
        for m2 in matches2to1:
            if m1.queryIdx == m2.trainIdx and m2.queryIdx == m1.trainIdx:
                good_matches.append(m1)
                break
    return good_matches

def sift_descriptor(image, keypoints):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(gray, keypoints)
    return descriptors

def brisk_descriptor(image, keypoints):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brisk = cv2.BRISK_create()
    _, descriptors = brisk.compute(gray, keypoints)
    return descriptors

def reScaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent // 100)
    height = int(frame.shape[0] * percent // 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def main():
    imageoriginal = cv2.imread('eggimg1.jpeg')
    image = reScaleFrame(imageoriginal, percent=50)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    harris_result = harris_corner_detector(image.copy())
    cv2.imshow('Harris Corners', harris_result)
    cv2.waitKey(0)

    dog_result = dog_detector(image.copy())
    cv2.imshow('DOG', dog_result)
    cv2.waitKey(0)

    mser_result = mser_detector(image.copy())
    cv2.imshow('MSER', mser_result)
    cv2.waitKey(0)

    shi_tomasi_result = shi_tomasi_corner_detection(gray.copy())
    cv2.imshow('Shi-Tomasi Corners', shi_tomasi_result)
    cv2.waitKey(0)

    sift_result, sift_keypoints, sift_descriptors = sift_detector(gray.copy())
    cv2.imshow('SIFT Keypoints', sift_result)
    cv2.waitKey(0)

    fast_result, fast_keypoints = fast_detector(gray.copy())
    cv2.imshow('FAST Keypoints', fast_result)
    cv2.waitKey(0)

    orb_result, orb_keypoints, orb_descriptors = orb_detector(gray.copy())
    cv2.imshow('ORB Keypoints', orb_result)
    cv2.waitKey(0)

    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    sift_descriptors = sift_descriptor(image, keypoints)

    brisk = cv2.BRISK_create()
    keypoints = brisk.detect(gray, None)
    brisk_descriptors = brisk_descriptor(image, keypoints)

    brisk_result, brisk_keypoints, brisk_descriptors = brisk_detector(gray.copy())
    cv2.imshow('BRISK Keypoints', brisk_result)
    cv2.waitKey(0)

    kaze_result, kaze_keypoints, kaze_descriptors = kaze_detector(gray.copy())
    cv2.imshow('KAZE Keypoints', kaze_result)
    cv2.waitKey(0)

    akaze_result, akaze_keypoints, akaze_descriptors = akaze_detector(gray.copy())
    cv2.imshow('AKAZE Keypoints', akaze_result)
    cv2.waitKey(0)

    desc1 = sift_descriptors
    desc2 = brisk_descriptors
    mask = np.ones(len(desc1), dtype=bool)
    matches_bf = bf_matcher(desc1, desc2)
    matches_flann = flann_matcher(desc1, desc2)
    matches_brute_force = brute_force_matcher(desc1, desc2)
    matches_flann_based = flann_based_matcher(desc1, desc2)
    matches_knn = knn_matcher(desc1, desc2)
    matches_radius = radius_matcher(desc1, desc2)
    matches_ratio = ratio_matcher(desc1, desc2)
    matches_cross_check = cross_check_matcher(desc1, desc2)
    matches_ransac = ransac_matcher(desc1, desc2, sift_keypoints, brisk_keypoints, mask)
    matches_cross_check = brute_force_cross_check_matcher(desc1, desc2)

    img_matches_bf = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_bf, None)
    cv2.imshow("BF Matcher", img_matches_bf)
    cv2.waitKey(0)

    img_matches_flann = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_flann, None)
    cv2.imshow("FLANN Matcher", img_matches_flann)
    cv2.waitKey(0)

    img_matches_brute_force = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_brute_force, None)
    cv2.imshow("Brute Force Matcher", img_matches_brute_force)
    cv2.waitKey(0)

    img_matches_flann_based = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_flann_based, None)
    cv2.imshow("FLANN-Based Matcher", img_matches_flann_based)
    cv2.waitKey(0)

    img_matches_knn = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_knn, None)
    cv2.imshow("KNN Matcher", img_matches_knn)
    cv2.waitKey(0)

    img_matches_radius = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_radius, None)
    cv2.imshow("Radius Matcher", img_matches_radius)
    cv2.waitKey(0)

    img_matches_ratio = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_ratio, None)
    cv2.imshow("Ratio Matcher", img_matches_ratio)
    cv2.waitKey(0)

    img_matches_cross_check = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_cross_check, None)
    cv2.imshow("Cross Check Matcher", img_matches_cross_check)
    cv2.waitKey(0)

    img_matches_ransac = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_ransac, None)
    cv2.imshow("RANSAC Matcher", img_matches_ransac)
    cv2.waitKey(0)

    img_matches_cross_check = cv2.drawMatches(image, sift_keypoints, image, brisk_keypoints, matches_cross_check, None)
    cv2.imshow("Brute-Force Cross-Check Matcher", img_matches_cross_check)
    cv2.waitKey(0)


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()