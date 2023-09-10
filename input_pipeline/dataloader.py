from torch.utils.data import DataLoader


def threaded_loader(loader, iscuda, threads, batch_size=1, shuffle=True):
    """ Get a data loader, given the dataset and some parameters.

        Parameters
        ----------
        loader : object[i] returns the i-th training example.

        iscuda : bool

        batch_size : int

        threads : int

        shuffle : bool

        Returns
        -------
            a multi-threaded pytorch loader.
        """
    return DataLoader(
        loader,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=threads,
        pin_memory=iscuda)
