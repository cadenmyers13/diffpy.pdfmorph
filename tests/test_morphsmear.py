#!/usr/bin/env python


import os

import numpy
import pytest

from diffpy.morph.morphs.morphsmear import MorphSmear

# useful variables
thisfile = locals().get("__file__", "file.py")
tests_dir = os.path.dirname(os.path.abspath(thisfile))
# testdata_dir = os.path.join(tests_dir, 'testdata')


class TestMorphSmear:
    @pytest.fixture
    def setup(self):
        self.smear = 0.1
        rmax = 10
        self.r0 = 7 * numpy.pi / 22.0 * rmax / 2
        self.x_morph = numpy.arange(0.01, rmax, 0.01)
        self.y_morph_gaussian = numpy.exp(
            -0.5 * ((self.x_morph - self.r0) / self.smear) ** 2
        )
        self.y_morph_lorentzian = self.smear / (
            (self.x_morph - self.r0) ** 2 + self.smear**2
        )
        self.x_target = self.x_morph.copy()
        self.y_target = self.x_target.copy()
        return

    def test_morph(self, setup):
        """Check MorphSmear.morph()"""
        # Test Gaussian (default) morph is applied correctly
        morph = MorphSmear()
        morph.smear = 0.15
        morph.smear_func = None

        x_morph, y_morph, x_target, y_target = morph(
            self.x_morph, self.y_morph_gaussian, self.x_target, self.y_target
        )

        # Target should be unchanged
        assert numpy.allclose(self.y_target, y_target)

        # Compare to broadened Gaussian
        sigbroad = (self.smear**2 + morph.smear**2) ** 0.5
        ysmear = numpy.exp(-0.5 * ((self.x_morph - self.r0) / sigbroad) ** 2)
        ysmear *= self.smear / sigbroad

        assert numpy.allclose(ysmear, y_morph)

        # Test Lorentzian morph is applied correctly
        morph = MorphSmear()
        morph.smear = 0.15
        morph.smear_func = "lorentzian"

        x_morph, y_morph, x_target, y_target = morph(
            self.x_morph, self.y_morph_lorentzian, self.x_target, self.y_target
        )

        # Target should be unchanged
        assert numpy.allclose(self.y_target, y_target)

        # Compare to broadened Lorentzian
        sigbroad = self.smear + abs(morph.smear)
        ysmear = sigbroad / ((self.x_morph - self.r0) ** 2 + sigbroad**2)
        # Need to scale arbitrarily due to non-exponential tail decay
        ysmear *= max(y_morph) / max(ysmear)

        # Set higher atol since Lorentzian tails will be off
        assert numpy.allclose(ysmear, y_morph, atol=0.005)
        return
