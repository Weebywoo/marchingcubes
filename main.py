import marchingcubes


def main() -> None:
    vertices, faces = marchingcubes.construct(25, sample_points=3, scale=0.15)
    marchingcubes.plot(vertices, faces)


if __name__ == "__main__":
    main()
