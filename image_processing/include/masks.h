#pragma once

namespace Mask {

    class BaseMask {
    public:
        virtual ~BaseMask() = default;
        virtual int getWidth() const = 0;
        virtual int getHeight() const = 0;
        virtual int getCenterRow() const = 0;
        virtual int getCenterColumn() const = 0;
        virtual double getFilterFactor() const = 0;
        virtual const double* getData() const = 0;
    };

    class GaussianBlur3 : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 16;
        double mask[9];

    public:

        GaussianBlur3() {
            double filter[9] = {
            1.0, 2.0, 1.0,
            2.0, 4.0, 2.0,
            1.0, 2.0, 1.0
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class GaussianBlur5 : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 273;
        double mask[25];
    public:
        GaussianBlur5() {
            double filter[25] = {
            1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
            4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
            7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
            4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
            1.0f,  4.0f,  7.0f,  4.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class SharpenMask : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 8.0;
        double mask[25];
    public:
        SharpenMask() {
            double filter[25] = {
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
            -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
            -1.0f,  2.0f,  8.0f,  2.0f, -1.0f,
            -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class VertEdgeDetect : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 1.0;
        double mask[25];
    public:
        VertEdgeDetect() {
            double filter[25] = {
            0,  0, -1.0f,  0,  0,
            0,  0, -1.0f,  0,  0,
            0,  0,  4.0f,  0,  0,
            0,  0, -1.0f,  0,  0,
            0,  0, -1.0f,  0,  0
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class EdgeSharpen : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        double mask[9];

    public:

        EdgeSharpen() {
            double filter[9] = {
            1.0f,  1.0f, 1.0f,
            1.0f, -7.0f, 1.0f,
            1.0f,  1.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class Emboss3D : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        double mask[9];

    public:

        Emboss3D() {
            double filter[9] = {
            2.0f,  0.0f,  0.0f,
            0.0f, -1.0f,  0.0f,
            0.0f,  0.0f, -1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class EdgeSobelX : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        double mask[9];

    public:

        EdgeSobelX() {
            double filter[9] = {
                -1.0f, 0.0f, 1.0f,
                -2.0f, 0.0f, 2.0f,
                -1.0f, 0.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class EdgeSobelY : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        double mask[9];

    public:

        EdgeSobelY() {
            double filter[9] = {
                -1.0f, -2.0f, -1.0f,
                0.0f,  0.0f,  0.0f,
                1.0f,  2.0f,  1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

    class BoxBlur : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 9.0;
        double mask[9];

    public:

        BoxBlur() {
            double filter[9] = {
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask;
        }
    };

}