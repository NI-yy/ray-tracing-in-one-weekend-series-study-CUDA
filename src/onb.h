#ifndef ONB_H
#define ONB_H

class onb {
public:
    onb(const vec3& n) {
        axis[2] = unit_vector(n); // 法線をw軸（axis[2]）とする
        // w軸と平行にならないように補助ベクトルaを選択
        vec3 a = (std::fabs(axis[2].x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        axis[1] = unit_vector(cross(axis[2], a)); // v軸
        axis[0] = cross(axis[2], axis[1]);        // u軸
    }

    const vec3& u() const { return axis[0]; }
    const vec3& v() const { return axis[1]; }
    const vec3& w() const { return axis[2]; }

    vec3 transform(const vec3& v) const {
        // 基底座標からローカル空間（ワールド空間内の向き）へ変換
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }

private:
    vec3 axis[3];
};

#endif