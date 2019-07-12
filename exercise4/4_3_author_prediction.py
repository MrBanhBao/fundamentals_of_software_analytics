import sys
import numpy as np
import git
import pandas as pd
import os
import nltk
import time
from sklearn.manifold import TSNE
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


def main():
    # cwd = os.getcwd()

    # !!! INIT PHASE !!! Took too long, that's why we saved our results
    cwd = os.getcwd()
    # start_time = time.time()
    # df = create_commit_df(cwd)
    # df.to_hdf('100_days_commit_alphaonly.h5', key='commit')
    # print("--- %s seconds ---" % (time.time() - start_time))

    for stdin in sys.stdin:
        file = str.strip(stdin)
        with open(os.path.join(cwd, file)) as f:
            source_code = f.read()
            sc_tokens = tokenize_sc(source_code)

        df = pd.read_hdf('../../100_days_commits.h5')  # load data from init phase
        vocab = create_vocab(df.tokens, top=256)
        df = df.groupby('author.name', as_index=False).agg({'tokens': sum})
        author_bow_df = create_bow_df(df, vocab)
        vecs = [np.array(list(bow.values())) for bow in author_bow_df.bow.values]

        sc_bow = create_bow(sc_tokens, vocab)
        sc_vec = [np.array(list(sc_bow.values()))]

        tsne = TSNE(n_components=2, random_state=128)
        vecs_2d = tsne.fit_transform(vecs+sc_vec)

        vor = Voronoi(vecs_2d[:-1])
        regions, vertices = voronoi_finite_polygons_2d(vor)

        offset = 10
        point = Point(np.array(vecs_2d[-1]))

        voronoi_plot_2d(vor)
        for i, region in enumerate(regions):
            polygon_points = vertices[region]
            polygon = Polygon(polygon_points)
            if polygon.contains(point):
                print(author_bow_df.iloc[i]['author.name'])
                plt.fill(*zip(*polygon_points), alpha=0.4)
                break

        plt.plot(vecs_2d[:-1, 0], vecs_2d[:-1, 1], 'ko')
        plt.xlim(vor.min_bound[0] - offset, vor.max_bound[0] + offset)
        plt.ylim(vor.min_bound[1] - offset, vor.max_bound[1] + offset)
        plt.plot(point.x, point.y, 'rs')
        plt.show()


def create_commit_df(git_dir):
    repo = git.Repo(git_dir)
    data = [{
        'hexsha': commit.hexsha,
        'author.name': commit.author.name,
        'author.email': commit.author.email,
        'committed_date': pd.to_datetime(commit.committed_date, unit='s'),
        'commit_message': commit.message,
        'tokens': tokenize_diff(commit)
    } for commit in repo.iter_commits('master',  since='100.days.ago')]

    return pd.DataFrame(data)


def tokenize_diff(commit, alpha_only=True):
    words = []

    parent = commit.parents[0] if commit.parents else None
    diffs = [diff for diff in commit.diff(parent, create_patch=True)]

    for diff in diffs:
        try:
            diff_text = diff.diff.decode("utf-8")
            tokens = nltk.word_tokenize(diff_text)
            if alpha_only:
                words += [word.lower() for word in tokens if word.isalpha()]
            else:
                words += [word.lower() for word in tokens]
        except Exception as e:
            print(e, 'at Commit', commit.hexsha)
            continue

    return words


def tokenize_sc(sc, alpha_only=True):
    tokens = nltk.word_tokenize(sc)
    if alpha_only:
        return [word.lower() for word in tokens if word.isalpha()]
    else:
        return [word.lower() for word in tokens]


def create_bow(tokens, vocab):
    bow = get_empty_bow(vocab)

    for token in tokens:
        if token in list(bow.keys()):
            bow[token] += 1
    return bow


def create_vocab(tokens, top=256):
    word_freqs = nltk.FreqDist(np.concatenate(tokens))
    top_words_freqs = word_freqs.most_common(top)
    top_words = [word_freq[0] for word_freq in top_words_freqs]
    return sorted(top_words)


def create_bow_df(df, vocab):
    bows = []
    author_names = []
    for index, commit in df.iterrows():
        bow = get_empty_bow(vocab)
        author_name = commit['author.name']

        tokens = commit.tokens
        for token in tokens:
            if token in list(bow.keys()):
                bow[token] += 1

        author_names.append(author_name)
        bows.append(bow)

    return pd.DataFrame({'author.name': author_names, 'bow': bows})


def get_empty_bow(vocabs):
    word_vec = {}
    for vocab in vocabs:
        word_vec[vocab] = 0

    return word_vec


# From: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


if __name__ == '__main__':
    main()

