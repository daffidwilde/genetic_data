""" A collection of useful functions for various processes within the GA. """


def get_family_counts(metadata, families):
    """ Get the number of instances in `metadata` that belong to each family in
    `families`. """

    return {
        family: sum([pdf.name == family.name for pdf in metadata])
        for family in families
    }
